
# %%
import torch as t
from torch import Tensor
from typing import Optional, Union, Dict, Callable, Optional, List
from typing_extensions import Literal
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils
import itertools
from functools import partial
from tqdm.auto import tqdm
from jaxtyping import Float, Int 


# %%


def hook_fn_generic_patching(activation: Float[Tensor, "..."], hook: HookPoint, cache: ActivationCache) -> Float[Tensor, "..."]:
    '''Patches entire tensor of activations, from corresponding values in cache.'''
    activation[:] = cache[hook.name][:]
    return activation

def hook_fn_generic_caching(activation: Float[Tensor, "..."], hook: HookPoint, name: str = "activation") -> Float[Tensor, "..."]:
    '''Stores activations in hook context.'''
    hook.ctx[name] = activation
    return activation

def hook_fn_generic_patching_from_context(activation: Float[Tensor, "..."], hook: HookPoint, name: str = "activation") -> Float[Tensor, "..."]:
    '''Patches activations from hook context, if they are there.'''
    if name in hook.ctx:
        activation[:] = hook.ctx[name][:]
        hook.clear_context()
    return activation


class Component:
    '''
    Returns a component in a nice way, i.e. without requiring messy dicts.

    This class contains the following:
        Convenient ways to define components (see examples below)
        Error checking (making sure you don't define components incorrectly)
        Methods to return hook functions for stuff like patching

    We specify a component using:
        *args (the thing we'd index cache with)
        **kwargs (can be seq_pos, head or neuron: used to override the default of patching all values at once)

    Examples:
        Component("q", 0, head=1)
        Component("resid_pre", 1, seq_pos=[3, 4])
        Component("pre", -1, neuron=1337)
    '''
    def __init__(self, *args, **kwargs):
        self.args = args
        self.component_name = args[0]
        self.activation_name = utils.get_act_name(*args)
        assert all([k in ["seq_pos", "head", "neuron"] for k in kwargs.keys()]), "Error: kwargs must be one of 'seq_pos', 'head', or 'neuron'."
        self.seq_pos = kwargs.get("seq_pos", None)
        self.head = kwargs.get("head", None)
        self.neuron = kwargs.get("neuron", None)

        # Check if head dimension is appropriate
        if not any(name in self.activation_name for name in ["q", "k", "v", "z", "pattern", "attn_scores"]):
            assert self.head is None, f"Can't specify `head` for activation {self.activation_name}."

        # Check if neuron dimension is appropriate
        if not any(name in self.activation_name for name in ["pre", "post"]):
            assert self.neuron is None, f"Can't specify `neuron` for activation {self.activation_name}."


    def check_and_fix_sender_component(self):
        '''
        Raises an error if this isn't a valid sender component.

        Not sure exactly what a valid sender component would be, but this seemed like a good starting point.
        '''
        valid_sender_components = ["v", "z", "attn_out", "post", "mlp_out", "resid_pre", "resid_post", "resid_mid"]
        assert self.component_name in valid_sender_components, f"Error: component {self.component_name} is not a valid sender component. Valid sender components are: {valid_sender_components}"

    
    def check_and_fix_receiver_component(self):
        '''
        Checks if this is a valid receiver component.

        We need to make sure it's not "attn_out" or "mlp_out" (because if it's a receiver then caching needs to happen before patching in step 2).
        But rather than raising an error in this case, we can just change the patching to be at "z" or "post" instead.
        '''
        if self.component_name == "attn_out":
            self.component_name = "z" # This is causally the same as "change the attention head's output", but it won't screw with the caching step
        elif self.component_name == "mlp_out":
            self.component_name = "post" # This is causally the same as "change the MLP's output", but it won't screw with the caching step


    def get_patching_hook_fn(self, cache: ActivationCache) -> Callable:
        '''
        Returns a hook function for doing patching according to this component.

        This is for step 2 of path patching (i.e. where we're patching the output of sender components), so we
        assume that the component name is one of (z, attn_out, post, or mlp_out).
        '''
        def hook_fn(activations: Float[Tensor, "..."], hook: HookPoint) -> Float[Tensor, "..."]:
            # Define an index for slicing (by default slice(None), which is equivalent to [:])
            idx = [slice(None) for _ in range(activations.ndim)]

            # Check if we need to patch by head, and if so then check where the head dim is
            if self.head is not None:
                assert self.component_name in ["z", "v"], "Only places it's valid to patch by head are 'z' and 'v'."
                idx[-2] = self.head
            # Check if we need to patch by neuron, and if so then check where the neuron dim is
            if self.neuron is not None:
                assert self.component_name == "post", "Only place it's valid to patch by neuron is 'post'."
                idx[-1] = self.neuron
            # Lastly, we might also need to patch by sequence position (which is first dimension for all of these components)
            if self.seq_pos is not None:
                idx[1] = self.seq_pos

            # Now, patch the values in our activations tensor, and return the new activation values
            activations[idx] = cache[hook.name][idx]
            return activations

        return hook_fn

    
    def __repr__(self):
        return f"Component({self.args}, seq_pos={self.seq_pos}, head={self.head}, neuron={self.neuron})"



def product_with_args_kwargs(*args, **kwargs):
    '''
    Helper function which generates an iterable from args and kwargs.

    For example, running the following:
        product_with_args_kwargs([1, 2], ['a', 'b'], key1=[True, False])
    gives us:
        ((1, 'a'), {'key1', True})
        ((1, 'a'), {'key1', False})
        ((1, 'b'), ('key1', True))
        ...
        ((2, 'b'), {'key1', False})
    '''
    # Generate the product of args
    args_product = list(itertools.product(*args))
    
    # Generate the product of kwargs values
    kwargs_product = list(itertools.product(*kwargs.values()))
    
    # Combine args_product with each dict from kwargs_product
    result = []
    for args_values in args_product:
        for kwargs_values in kwargs_product:
            # Construct dict from keys and current values
            kwargs_dict = dict(zip(kwargs.keys(), kwargs_values))
            # Append the tuple with current args_values and kwargs_dict
            result.append((args_values, kwargs_dict))

    return result




class MultiComponent:
    '''
    Class for defining a list of components.

    In other words, we specify a few arguments, and this thing gives us a list of Component objects.

    Args:
        components
            name of component (or list of componnet names) which we iterate over
        seq_pos
            Is `None` by default, but can also be `"each"`, or it can be specified as an int or list of ints.

    Examples:
        MultiComponent(component_names=["q", "k", "v"], seq_pos="each")
            - this is how we might iterate over receiver components
            - it'll give us a dict with keys ("q", "k", "v") and values = tensors of shape (seq_pos, layers, heads)
            - each element would be the result of patching from some fixed sender to a different receiver

        MultiComponent(component_names=["attn_out", "mlp_out", "resid_pre"])
            - this is how we might iterate over sender components
            - it'll give us a dict with keys ("attn_out", "mlp_out", "resid_pre") and values = 1D tensors of shape (layers,)

        MultiComponent(component_names="attn_out", seq_pos="each")
            - this is how we might iterate over sender components
            - it'll give us a tensor of shape (layers, seq_pos)

    Note:
        If you want to iterate over each attn head, use 'z' (or something with a head dim). If you want to get all heads simultaneously, use 'attn_out'.
        If you want to iterate over each neuron, use 'post'. If you want to get all heads simultaneously, use 'attn_out'.
    '''
    def __init__(
        self, 
        component_names: Union[str, List[str]], 
        seq_pos: Optional[Union[Literal["each"], int, List[int]]] = None
    ):
        # Get component_names into a string, for consistency
        self.component_names = [component_names] if isinstance(component_names, str) else component_names

        # Figure out what the shape of the output will be (i.e. for our path patching iteration)
        self.shapes = {}
        for component in self.component_names:
            # Everything has a "layer" dim
            self.shapes[component] = ["layer"]
            # Add "seq_pos", if required
            if seq_pos == "each": self.shapes[component].append("seq_pos")
            # Add "head", if required
            if component in ["z", "q", "k", "v", "attn_scores", "pattern"]:
                self.shapes[component].append("head")


    def get_component_dict(self, model: HookedTransformer, logits: Float[Tensor, "batch seq_pos"]) -> Dict[str, List[Component]]:
        '''
        This is how we get the actual list of components (i.e. `Component` objects) which we'll be iterating through.

        We need `model` and `logits` to do it, because we need to know the shapes of the components (e.g. how many layers
        or sequence positions, etc).
        '''
        # Get a dict we can use to convert from shape names into actual shape values
        shape_values_dict = {"seq_pos": logits.shape[1], "layer": model.cfg.n_layers, "head": model.cfg.n_heads, "neuron": model.cfg.d_mlp}

        # Get a dictionary to store the components (i.e. this will be a list of `Component` objects for each component name)
        # Also, get dict to store the actual shape values for each component name
        self.components_dict = {}
        self.shape_values = {}

        # Fill in self.shape_values with the actual values, and self.components_dict using itertools.product
        # Note, this is a bit messy (e.g. component name and layer are *args, but head/neuron/seq_pos are **kwargs)
        for component_name, shape_names in self.shapes.items():
            # Get shape values
            shape_values = [shape_values_dict[shape_name] for shape_name in shape_names]
            # Get iterable, using a helper function defined earlier (using layers as *args and everything else as **kwargs)
            shape_args = []
            shape_kwargs = {}
            for shape_name, shape_value in zip(shape_names, shape_values):
                if shape_name == "layer": shape_args.append(range(shape_value))
                elif shape_name in ["head", "neuron", "seq_pos"]: shape_kwargs[shape_name] = range(shape_value)

            self.shape_values[component_name] = shape_values
            self.components_dict[component_name] = product_with_args_kwargs(*shape_args, **shape_kwargs)
            self.components_dict[component_name] = [Component(component_name, *args, **kwargs) for args, kwargs in self.components_dict[component_name]]

        return self.components_dict




def _path_patch_single(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    sender_components: Union[Component, List[Component]],
    receiver_components: Union[Component, List[Component]],
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], Float[Tensor, ""]],
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    sender_seq_pos: Optional[Union[int, List[int]]] = None,
    receiver_seq_pos: Optional[Union[int, List[int]]] = None,
    direct_includes_mlps: bool = True,
) -> Float[Tensor, ""]:
    '''
    Function which gets repeatedly called when doing path patching via the main `path_patch` function.

    This shouldn't be called directly by the user.
    '''
    # Call this at the start, just in case! This also clears context by default
    model.reset_hooks()

    # Turn the components into a list of components (for consistency)
    if isinstance(receiver_components, Component):
        receiver_components = [receiver_components]
    if isinstance(sender_components, Component):
        sender_components = [sender_components]

    # Check the components are valid
    # Also, if seq_pos isn't supplied for a single component, we override it with the `sender_seq_pos` or `receiver_seq_pos` argument
    for component in sender_components:
        if component.seq_pos is None: component.seq_pos = sender_seq_pos
        component.check_and_fix_sender_component()
    for component in receiver_components:
        if component.seq_pos is None: component.seq_pos = receiver_seq_pos
        component.check_and_fix_receiver_component()
    


    # ========== Step 2 ==========
    # Run model on orig with sender components patched from new and all other components frozen. Cache the receiver components.

    # We need to define three sets of hook functions: for freezing heads, for patching senders (which override freezing), and for caching receivers (before freezing)
    hooks_for_freezing = []
    hooks_for_caching_receivers = []
    hooks_for_patching_senders = []

    # Get all the hooks we need for freezing heads (and possibly MLPs)
    # (note that these are added at "z" and "post", because if they were added at "attn_out" or "mlp_out" then we might not be able to override them with the patching hooks)
    hooks_for_freezing.append((lambda name: name.endswith("z"), partial(hook_fn_generic_patching, cache=orig_cache)))
    if not direct_includes_mlps:
        hooks_for_freezing.append((lambda name: name.endswith("post"), partial(hook_fn_generic_patching, cache=orig_cache)))

    # Get all the hooks we need for patching heads (and possibly MLPs)
    for component in sender_components:
        hooks_for_patching_senders.append((component.activation_name, component.get_patching_hook_fn(new_cache)))

    # Get all the hooks we need for caching receiver components
    for component in receiver_components:
        hooks_for_caching_receivers.append((component.activation_name, partial(hook_fn_generic_caching, name="receiver_activations")))

    # Now add all the hooks in order. Note that patching should override freezing, and caching should happen before both.
    model.run_with_hooks(
        orig_input,
        return_type=None,
        fwd_hooks=hooks_for_caching_receivers + hooks_for_freezing + hooks_for_patching_senders,
        clear_contexts=False # This is the default anyway, but just want to be sure!
    )
    # Result - we've now cached the receiver components (i.e. stored them in the appropriate hook contexts)



    # ========== Step 3 ==========
    # Run model on orig with receiver components patched from previously cached values.

    # We do this by just having a single hook function: patch from the hook context, if we added it to the context (while caching receivers in the previous step)
    logits = model.run_with_hooks(
        orig_input,
        return_type="logits",
        fwd_hooks=[(lambda name: True, partial(hook_fn_generic_patching_from_context, name="receiver_activations"))],
    )
    return patching_metric(logits)





def path_patch(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    sender_components: Union[MultiComponent, Component, List[Component]],
    receiver_components: Union[MultiComponent, Component, List[Component]],
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], Float[Tensor, ""]],
    orig_cache: Optional[ActivationCache] = None,
    new_cache: Optional[ActivationCache] = None,
    sender_seq_pos: Optional[Union[int, List[int]]] = None,
    receiver_seq_pos: Optional[Union[int, List[int]]] = None,
    direct_includes_mlps: bool = True,
    verbose: bool = False,
) -> Float[Tensor, "..."]:
    '''
    Performs a single instance / multiple instances of path patching, from sender component(s) to receiver component(s).

    Note, I'm using orig and new in place of clean and corrupted to avoid ambiguity. In the case of noising algs (which patching usually is),
    orig=clean and new=corrupted.

    ===============================================================
    How we perform a single instance of path patching:
    ===============================================================

    Path patching algorithm:
        (1) Gather all activations on orig and new distributions.
        (2) Run model on orig with sender components patched from new and all other components frozen. Cache the receiver components.
        (3) Run model on orig with receiver components patched from previously cached values.

    Step (1) is done here. Steps (2) and (3) are done in the backend function `_path_path_single_instance`.
        
    Args:
        model:
            The model we patch with
        
        orig_input:
            The original input to the model (string, list of strings, or tensor of tokens)
        
        new_input:
            The new input to the model (string, list of strings, or tensor of tokens)
            i.e. we're measuring the effect of changing the given path from orig->new

        sender_components:
            The components in the path that come first (i.e. we patch the path from sender to receiver).
            This is given as a `Component` instance, or list of `Component` instances. See the `Component` class for more details.
            Note, if it's a list, they are all treated as senders (i.e. a single value is returned), rather than one-by-one.
        
        receiver_components:
            The components in the path that come last (i.e. we patch the path from sender to receiver).
            This is given as a `Component` instance, or list of `Component` instances. See the `Component` class for more details.
            Note, if it's a list, they are all treated as receivers (i.e. a single value is returned), rather than one-by-one.
        
        patching_metric:
            Should take in a tensor of logits, and output a scalar tensor.
            This is how we calculate the value we'll return.

        verbose: 
            Whether to print out extra info (in particular, about the shape of the final output).

    Returns:
        Scalar tensor (i.e. containing a single value).

    ===============================================================
    How we perform multiple instances of path patching:
    ===============================================================

    We can also do multiple instances of path patching in sequence, i.e. we fix our sender component(s) and iterate over receiver components,
    or vice-versa.

    The way we do this is by using a `MultiComponent` instance, rather than a `Component` instance. For instance, if we want to fix receivers
    and iterate over senders, we would use a `MultiComponent` instance for receivers, and a `Component` instance for senders.

    See the `MultiComponent` class for more info on how we can specify multiple components.

    Returns:
        Dictionary of tensors, keys are the component names of whatever we're iterating over.
        For instance, if we're fixing a single sender head, and iterating over (mlp_out, attn_out) for receivers, then we'd return a dict 
        with keys "mlp_out" and "attn_out", and values are the tensors of patching metrics for each of these two receiver components.
    '''

    # Make sure we aren't iterating over both senders and receivers
    assert not all([isinstance(receiver_components, MultiComponent), isinstance(sender_components, MultiComponent)]), "Can't iterate over both senders and receivers!"

    # ========== Step 1 ==========
    # Gather activations on orig and new distributions (we only need attn heads and possibly MLPs)
    # This is so that we can patch/freeze during step 2
    if new_cache is None:
        # We get logits cause it's useful for the shape (input might not be token ids)
        logits, new_cache = model.run_with_cache(new_input, return_type="logits")
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_input, return_type=None)

    # Get most of the arguments we'll be using for the backend patching function
    kwargs = dict(
        model=model,
        orig_input=orig_input,
        sender_components=sender_components,
        receiver_components=receiver_components,
        patching_metric=patching_metric,
        orig_cache=orig_cache,
        new_cache=new_cache,
        sender_seq_pos=sender_seq_pos,
        receiver_seq_pos=receiver_seq_pos,
        direct_includes_mlps=direct_includes_mlps,
    )

    # If we're fixing sender(s), and iterating over receivers:
    if isinstance(receiver_components, MultiComponent):
        results_dict = dict()
        receiver_components_dict = receiver_components.get_component_dict(model, logits)
        for receiver_component_name, receiver_component_list in receiver_components_dict.items():
            results_dict[receiver_component_name] = []
            for receiver_component in tqdm(receiver_component_list, desc=f"Patching over receivers: {receiver_component_name}"):
                kwargs.update({"receiver_components": receiver_component})
                results_dict[receiver_component_name].append(_path_patch_single(**kwargs).item())
        if verbose:
            print("Fixing senders, iterating over receivers")
            for component_name, component_shape in receiver_components.shapes.items():
                component_shape_value = receiver_components.shape_values[component_name]
                print(f"results[{component_name}].shape = {', '.join(f'{s}={v}' for s, v in zip(component_shape, component_shape_value))}")
        return {component_name: t.tensor(results).reshape(receiver_components.shape_values[component_name]) for component_name, results in results_dict.items()}
    # If we're fixing receiver(s), and iterating over senders:
    elif isinstance(sender_components, MultiComponent):
        results_dict = dict()
        sender_components_dict = sender_components.get_component_dict(model, logits)
        for sender_component_name, sender_component_list in sender_components_dict.items():
            results_dict[sender_component_name] = []
            for sender_component in tqdm(sender_component_list, desc=f"Patching over senders: {sender_component_name}"):
                kwargs.update({"sender_components": sender_component})
                results_dict[sender_component_name].append(_path_patch_single(**kwargs).item())
        if verbose:
            print("Fixing receivers, iterating over senders")
            for component_name, component_shape in sender_components.shapes.items():
                component_shape_value = sender_components.shape_values[component_name]
                print(f"results[{component_name}].shape = ({', '.join(f'{s}={v}' for s, v in zip(component_shape, component_shape_value))})")
        return {component_name: t.tensor(results).reshape(sender_components.shape_values[component_name]) for component_name, results in results_dict.items()}
    # If we're not iterating over anything, i.e. it's just a single instance of path patching:
    else:
        return _path_patch_single(**kwargs)