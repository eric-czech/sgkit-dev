FROM jupyter/minimal-notebook:54462805efcb
ENV WORK_DIR=$HOME/work
RUN mkdir $WORK_DIR/repos $WORK_DIR/auth $WORK_DIR/data $WORK_DIR/logs

USER root

# Install hail
# See:
# - https://github.com/eric-czech/hail-experiments/blob/master/Dockerfile
# - https://hail.is/docs/0.2/getting_started.html
# - https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#add-a-python-3-x-environment
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/*
# Note that hail does not run on py3.8 @ TOW
RUN conda create --quiet --yes -p $CONDA_DIR/envs/hail python=3.7 ipython ipykernel && \
    conda clean --all -f -y
RUN $CONDA_DIR/envs/hail/bin/pip install hail
RUN $CONDA_DIR/envs/hail/bin/python -m ipykernel install --user --name=hail

# Install VS code-server
ARG VSCODE_VERSION=3.4.1
RUN cd /opt \
  && wget -q https://github.com/cdr/code-server/releases/download/${VSCODE_VERSION}/code-server-${VSCODE_VERSION}-linux-x86_64.tar.gz \
  && tar -xf code-server-${VSCODE_VERSION}-linux-x86_64.tar.gz \
  && rm code-server-${VSCODE_VERSION}-linux-x86_64.tar.gz \
  && chmod a+x code-server-${VSCODE_VERSION}-linux-x86_64/code-server \
  && ln -s /opt/code-server-${VSCODE_VERSION}-linux-x86_64/code-server /usr/local/bin/code-server \
  && code-server --install-extension ms-python.python \
  && code-server --install-extension njpwerner.autodocstring
COPY dev.code-workspace $WORK_DIR/
COPY start-all.sh start-vscode.sh /usr/local/bin/
RUN chmod a+rx /usr/local/bin/start-all.sh /usr/local/bin/start-vscode.sh

# Install misc system packages
RUN apt-get update && apt-get install -y openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Download/install target repos
ARG GIT_USERNAME=eric-czech
ARG GIT_USEREMAIL=eric.allen.czech@gmail.com
ARG SGKIT_REPO=eric-czech/sgkit.git
ARG SGKIT_PLINK_REPO=eric-czech/sgkit-plink.git
ENV SGKIT_REPO=$SGKIT_REPO
ENV SGKIT_PLINK_REPO=$SGKIT_PLINK_REPO
RUN cd $WORK_DIR/repos && \
  git clone https://github.com/$SGKIT_REPO && \
  git clone https://github.com/$SGKIT_PLINK_REPO
#ENV PYTHONPATH="${PYTHONPATH}:$WORK_DIR/repos/sgkit"
ENV PYTHONPATH="${PYTHONPATH}:$WORK_DIR/repos/sgkit-plink"
#ENV PYLINTRC="$WORK_DIR/repos/public-data-source/.pylintrc"

# Install pysnptools separately (does not work as pip install with conda env update)
#RUN apt-get update && apt-get install -y g++
#RUN /conda/envs/rapids/bin/pip install --no-cache-dir pysnptools==0.4.11

# See: https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#using-pip-install-or-conda-install-in-a-child-docker-image
RUN cd $WORK_DIR/repos/sgkit && pip install -e .
RUN pip install \
  -r $WORK_DIR/repos/sgkit-plink/requirements.txt \
  -r $WORK_DIR/repos/sgkit-plink/requirements-dev.txt

# Ensure this always occurs last before user switch
RUN fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

USER $NB_UID

# Configure git + ssh
RUN mkdir $HOME/.ssh && chmod 700 $HOME/.ssh && \
  ssh-keyscan -t rsa github.com >> $HOME/.ssh/known_hosts
RUN cd $WORK_DIR/repos/sgkit && pre-commit install && \
  cd $WORK_DIR/repos/sgkit-plink && pre-commit install
RUN git config --global user.name $GIT_USERNAME && \
  git config --global user.email $GIT_USEREMAIL && \
  git config --global core.editor "vim"

# Modify source repository origins to use ssh on run (bind-mounted keys must be present)
CMD cd $WORK_DIR/repos/sgkit && git remote set-url origin git@github.com:$SGKIT_REPO && cd - && \
  cd $WORK_DIR/repos/sgkit-plink && git remote set-url origin git@github.com:$SGKIT_PLINK_REPO && cd - && \
  start-all.sh
