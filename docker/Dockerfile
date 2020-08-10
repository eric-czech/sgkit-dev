FROM jupyter/minimal-notebook:54462805efcb
ENV WORK_DIR=$HOME/work
RUN mkdir $WORK_DIR/repos $WORK_DIR/auth $WORK_DIR/data $WORK_DIR/logs

USER root

# Install bgen
# See: https://enkre.net/cgi-bin/code/bgen/dir?ci=trunk
# RUN apt-get update && \
#   apt-get install -y zlib1g-dev && \
#   rm -rf /var/lib/apt/lists/*
# RUN cd /opt && \
#   wget http://code.enkre.net/bgen/tarball/release/bgen.tgz && \
#   tar -xvf bgen.tgz && \
#   mv bgen.tgz bgen && \
#   cd bgen && \
#   ./waf configure && \
#   ./waf && \
#   ./build/test/unit/test_bgen > /dev/null

# Install regenie
# See: https://rgcgithub.github.io/regenie/install/
# There were no tags/releases at TOW so arbitrarily pick
# https://github.com/rgcgithub/regenie/commit/2e121b9756e600fb0f5c53192bbaba873a176e0c
# RUN cd $WORK_DIR/repos && \
#   git clone https://github.com/rgcgithub/regenie.git && \
#   cd regenie && \
#   git checkout 2e121b9756e600fb0f5c53192bbaba873a176e0c && \
#   sed -i 's@BGEN_PATH =@BGEN_PATH =/opt/bgen@' Makefile && \
#   make && \
#   ln -s $WORK_DIR/repos/regenie/regenie /usr/local/bin/regenie

# Install Hail
# See:
# - https://github.com/eric-czech/hail-experiments/blob/master/Dockerfile
# - https://hail.is/docs/0.2/getting_started.html
# - https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#add-a-python-3-x-environment
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/*
# Note that hail does not run on py3.8 @ TOW
RUN conda create --quiet --yes -p $CONDA_DIR/envs/hail python=3.7 ipython ipykernel pyarrow=0.15.1 fire && \
    conda clean --all -f -y
RUN $CONDA_DIR/envs/hail/bin/pip install hail==0.2.47
RUN $CONDA_DIR/envs/hail/bin/python -m ipykernel install --user --name=hail

# Install Glow
COPY environment-glow.yml /tmp/
RUN conda env create -p $CONDA_DIR/envs/glow -f /tmp/environment-glow.yml && \
    conda clean --all -f -y
RUN $CONDA_DIR/envs/glow/bin/pip install glow.py==0.5.0
RUN $CONDA_DIR/envs/glow/bin/python -m ipykernel install --user --name=glow

# Install VS code-server
ARG VSCODE_VERSION=3.2.0
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

# Install dev environment packages
# See: https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#using-pip-install-or-conda-install-in-a-child-docker-image
COPY environment.yml environment-dev.yml /tmp/
RUN conda env update -n base --file /tmp/environment.yml
RUN conda env update -n base --file /tmp/environment-dev.yml

# Install pysnptools separately (does not work as pip install with conda env update)
RUN pip install --no-cache-dir pysnptools==0.4.20

# Ensure this always occurs last before user switch
RUN fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

USER $NB_UID

ENV PYTHONPATH="${PYTHONPATH}:$WORK_DIR/repos/sgkit"
ENV PYTHONPATH="${PYTHONPATH}:$WORK_DIR/repos/sgkit-plink"
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

ARG SPARK_DRIVER_MEMORY=64g
ENV SPARK_DRIVER_MEMORY=$SPARK_DRIVER_MEMORY
# Set this as needed to avoid https://issues.apache.org/jira/browse/SPARK-29367
# with any pyspark 2.4.x + pyarrow >= 0.15.x
# See: https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x
ENV ARROW_PRE_0_15_IPC_FORMAT=1

CMD start-all.sh
