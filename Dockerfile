FROM pytorch/pytorch:latest
RUN apt-get update \
    && apt-get install -y curl

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

RUN pip install plotly torchtyping jupyterlab scikit-learn ipywidgets matplotlib openai
RUN pip install typeguard==2.13.3
COPY CircuitsVis CircuitsVis
RUN pip install CircuitsVis/python
COPY TransformerLens TransformerLens
RUN pip install -e ./TransformerLens
RUN pip install transformers==4.34

RUN git config --global user.email "oskar.hollinsworth@gmail.com" && \
    git config --global user.name "skar0" && \
    git config --global remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y yarn

RUN pip install -U kaleido

RUN apt install wget -y
RUN wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.focal_amd64.deb
RUN apt install -f ./wkhtmltox_0.12.6-1.focal_amd64.deb -y
RUN pip install imgkit
RUN pip install pytest transformers_stream_generator tiktoken

