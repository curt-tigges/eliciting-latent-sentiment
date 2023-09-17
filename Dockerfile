FROM pytorch/pytorch:latest
RUN pip install transformer-lens plotly torchtyping jupyterlab scikit-learn ipywidgets matplotlib openai
RUN pip install typeguard==2.13.3
RUN pip install -e CircuitsVis/python
RUN apt-get update \
    && apt-get install -y curl

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

RUN git config --global user.email "oskar.hollinsworth@gmail.com" && \
    git config --global user.name "skar0" && \
    git config --global remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"