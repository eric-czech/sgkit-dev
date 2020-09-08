# sgkit-dev

```bash

# Install gcloud + helm and authenticate for GS

# Create the Kubernetes cluster
gcloud container clusters create \
  --machine-type n1-standard-8 \
  --num-nodes 24 \
  --zone us-east-1c \
  --node-locations us-east-1c \
  --cluster-version latest \
  --scopes storage-rw \
  ukb-dask-1
  
# Launch the Dask cluster
helm install ukb-dask-helm-1 dask/dask 

# Open ports for UI and scheduler
kubectl port-forward --namespace default \
    svc/ukb-dask-helm-1-scheduler 8786:8786 &
kubectl port-forward --namespace default \
    svc/ukb-dask-helm-1-scheduler 80:80 &

# Launch Jupyter and client connection is automatic
export DASK_SCHEDULER_ADDRESS=tcp://127.0.0.1:8786
jupyter lab


```