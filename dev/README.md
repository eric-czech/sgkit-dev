

```
docker build -t sgkit-dev .
```

```
WORK_DIR=/home/jovyan/work
docker run --rm -ti \
-e GRANT_SUDO=yes --user=root \
-p 8898:8888 -p 8897:8887 \
-e JUPYTER_TOKEN=ItORmEnSTaTeNloRADHonisi \
-e VSCODE_TOKEN=ItORmEnSTaTeNloRADHonisi \
-e SPARK_DRIVER_MEMORY=64g \
-e JUPYTER_ENABLE_LAB=yes \
-v $HOME/repos/pystatgen/sgkit:$WORK_DIR/repos/sgkit \
-v $HOME/repos/pystatgen/sgkit-plink:$WORK_DIR/repos/sgkit-plink \
-v /data/disk1/dev:$WORK_DIR/data \
sgkit-dev
```
