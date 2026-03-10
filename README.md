# DGX Spark Cluster Bootstrap Guide (vLLM)

## Overview

This document describes how to bootstrap and run a **vLLM inference
cluster on a pair of NVIDIA DGX Spark systems** using the repository:

https://github.com/eugr/spark-vllm-docker

The repository provides a complete orchestration layer for:

-   building a vLLM Docker image
-   distributing it across nodes
-   launching a **Ray-backed distributed vLLM cluster**
-   running inference pipelines using **predefined YAML recipes**

Our internal setup consists of:

-   **2 DGX Spark nodes**
-   **1 GPU per node**
-   **RoCE / RDMA networking between nodes**
-   distributed inference via **Ray + vLLM**

This document assumes the use of **our internal fork of the
repository**, which already contains small patches and adjustments
required for our environment.

------------------------------------------------------------------------

# Network Configuration Background

## Initial Problem

When the DGX Spark nodes were first connected, the network interfaces
automatically configured **link-local IP addresses**:

    169.254.x.x

These addresses are automatically assigned when DHCP is unavailable.

While basic connectivity worked, this configuration caused several
issues:

-   RDMA / RoCE stack binding to the wrong interface
-   NCCL selecting non-optimal paths
-   degraded GPU-to-GPU throughput across nodes

This was observable in bandwidth tests such as:

    ib_write_bw

where performance did not match expected network throughput.

------------------------------------------------------------------------

## Solution: Dedicated Private Subnets

We resolved the issue by configuring **dedicated private subnets for the
RDMA interfaces**.

Example configuration:

Node 1 (Spark-9080)

    192.168.200.1
    192.168.201.1

Node 2 (Spark-dcb4)

    192.168.200.2
    192.168.201.2

This ensures that:

-   RDMA traffic uses deterministic interfaces
-   NCCL correctly binds to the RoCE devices
-   inter-node GPU communication achieves full bandwidth

------------------------------------------------------------------------

# Prerequisites

Before running the cluster, ensure the following prerequisites are
satisfied.

## 1. Passwordless SSH

The cluster scripts rely on **SSH-based orchestration** to start
containers on worker nodes.

Passwordless SSH must be configured **user-to-user** between all nodes.

Follow the original NVIDIA docs to set it up for your user (**Step 1** and **Step 4** only):

https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks

------------------------------------------------------------------------

# Image Build Status

The initial bootstrap phase has **already been completed**.

The following command was executed previously:

``` bash
./build-and-copy.sh
```

This process:

1.  Builds the `vllm-node` Docker image
2.  Copies the image to all cluster nodes
3.  Loads it into the Docker daemon on each node

Because the image is already present on both DGX Spark nodes, **this
step does not need to be repeated**.

You can verify the image exists locally with:

``` bash
docker images | grep vllm-node
```

Expected output:

    vllm-node   latest   <image-id>

------------------------------------------------------------------------

# Recipes

The repository provides **recipes** to simplify launching distributed
inference services.

A recipe is a YAML configuration that describes:

-   which model to serve
-   how to configure vLLM
-   required patches or mods
-   environment variables
-   GPU configuration
-   cluster execution mode

Recipes are stored in:

    recipes/

Example recipe:

    recipes/qwen35-4b.yaml

Using recipes is the **recommended way to launch the cluster**, because
they provide a complete, reproducible configuration.

Both sparks already export `HF_HOME=/opt/hf_shared` in `.bashrc`, so recipe launches
automatically use the default shared HuggingFace cache without any extra
setup.

If you want to use a different cache location for a specific session,
manually export `HF_HOME` before running the recipe.

------------------------------------------------------------------------

# Launching the Cluster with a Recipe

The easiest way to start the cluster is:

``` bash
./run-recipe.sh <recipe-name> -n <node1>,<node2>
```

Run this command on one node only. That node becomes the cluster head,
while the second node automatically joins as a worker.

Example for our two-node cluster:

``` bash
./run-recipe.sh qwen35-4b -n 192.168.200.1,192.168.200.2
```

This command will:

1.  Start the container on the **head node**
2.  Start containers on **worker nodes**
3.  Initialize the **Ray cluster**
4.  Launch the vLLM server using the parameters defined in the recipe

After startup, the vLLM API server will be accessible on the configured
port (defined in the recipe).

Example:

    http://<head-node>:8008

------------------------------------------------------------------------

# Verifying Cluster Status

You can check the Ray cluster status using:

``` bash
./launch-cluster.sh status
```

or inspect the container logs:

``` bash
docker logs -f vllm_node
```

------------------------------------------------------------------------

# Quick Inference Test Script

To quickly validate the loaded vLLM server, use:

    python/test.py

from this repository.

The script reads runtime parameters from:

    python/test.config.json

and sends a chat completion request to the server using your configured
OpenAI-compatible environment variables.

`python/test.py` no longer uses in-code defaults. You must provide a
JSON config via `--config`; otherwise the script exits with a config
error.

Run it from the repository root:

``` bash
python3 python/test.py --config python/test.config.json
```

You can also pass a custom config file:

``` bash
python3 python/test.py --config /path/to/config.json
```

------------------------------------------------------------------------

# Summary

This setup provides:

-   reproducible cluster launches
-   automated Docker orchestration
-   distributed vLLM inference
-   HuggingFace model caching
-   simplified deployment through recipes

With the networking and image build steps already completed, launching a
new model service typically requires only:

``` bash
./run-recipe.sh <recipe> -n 192.168.200.1,192.168.200.2
```

If a different cache is needed, manually export `HF_HOME` before launch.

which makes the system straightforward to operate for daily
experimentation and benchmarking.
