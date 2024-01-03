### Matrix Multiplication Function

This folder contains the code for *Matrix Multiplication* function for OpenFaaS deployment. For deployment instructions, kindly refer to OpenFaaS [tutorial](https://github.com/openfaas/workshop).


<br>

__NOTE:__ <br>
The current open-source version of OpenFaaS i.e. Community Edition, does not suppor scaling over `5` pods for a deployment and is meant for testing purposes only. To enable the scaling behaviour - <br> 

-   fork the open-source projects [`faas-netes`](https://github.com/openfaas/faas-netes) and [`faas-drl`](https://github.com/openfaas/faas)
-   edit the `DefaultMaxReplicas = 5` flag with desired value in both the projects, i.e., `gateway` and `faas-netes`
-   build the deployments via the `Dockerfile` by updating appropriate project paths
-   replace the running containers in OpenFaaS with the updated builds using Kubernetes API commands
