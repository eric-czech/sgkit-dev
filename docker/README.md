## Dev Image

Build:

```bash
docker build -t sgkit-dev .
```

Run:

```bash
WORK_DIR=/home/jovyan/work
docker run --rm -ti \
-e GRANT_SUDO=yes --user=root \
-p 8898:8888 -p 8897:8887 -p 8000:8000 \
-e JUPYTER_TOKEN=ItORmEnSTaTeNloRADHonisi \
-e VSCODE_TOKEN=ItORmEnSTaTeNloRADHonisi \
-e SPARK_DRIVER_MEMORY=64g \
-e JUPYTER_ENABLE_LAB=yes \
-v $HOME/.ssh:/home/jovyan/.ssh \
-v /data/disk1/dev:$WORK_DIR/data \
-v $HOME/repos/pystatgen/sgkit:$WORK_DIR/repos/sgkit \
-v $HOME/repos/pystatgen/sgkit-plink:$WORK_DIR/repos/sgkit-plink \
-v $HOME/repos/pystatgen/sgkit-bgen:$WORK_DIR/repos/sgkit-bgen \
-v $HOME/repos/pystatgen/sgkit-dev:$WORK_DIR/repos/sgkit-dev \
sgkit-dev
```
