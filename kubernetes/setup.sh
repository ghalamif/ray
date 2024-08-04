#!/bin/bash

gpu_nodes=("butter" "coffee")
namespace="kube-ray"
service_account="kuberay-service-account"
ray_cluster_value_path="kuube-ray/ray-cluster-values.yaml"

node_exists() {
    kubectl get node "$1" &> /dev/null
    return $?
}

node_has_label() {
    local node="$1"
    local label="$2"
    kubectl get node "$node" --show-labels | grep -q "$label"
    return $?
}

label_gpu_nodes() {
    local nodes=("$@")  # Accept nodes as an argument
    local gpu_label_key="gpu-node"
    local gpu_label_value="true"
    local gpu_label="${gpu_label_key}=${gpu_label_value}"

    # Iterate over the array and label each node if it exists and doesn't already have the label
    for node in "${nodes[@]}"; do
        if node_exists "$node"; then
            if node_has_label "$node" "$gpu_label"; then
                echo "Node '$node' already labeled with '$gpu_label'. Skipping."
            else
                echo "Node '$node' exists and is not labeled with '$gpu_label'. Adding label."
                kubectl label nodes "$node" $gpu_label
            fi
        else
            echo "Node '$node' does not exist. Skipping."
        fi
    done
}

nvidia_plugin_exists() {
    kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &> /dev/null
    return $?
}

install_nvidia_device_plugin() {
    if nvidia_plugin_exists; then
        echo "NVIDIA device plugin already installed. Skipping."
    else
        echo "Installing NVIDIA device plugin."
        kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
    fi
}


namespace_exists() {
    # Check if the namespace already exists and create it if it doesn't
    if kubectl get namespace $namespace &> /dev/null; then
        echo "Namespace $namespace already exists. Skipping."
    else
        echo "Creating namespace $namespace."
        kubectl create namespace $namespace
    fi
}

create_kube_ray_operator() {
    # Check if the namespace already exists and create it if it doesn't
    namespace_exists

    # Check if the service account already exists and create it if it doesn't
    if kubectl get serviceaccount $service_account -n $namespace &> /dev/null; then
        echo "Service account $service_account already exists. Skipping."
    else
        echo "Creating service account $service_account."
        kubectl apply -f service-account-for-ray.yaml
    fi

    # Create the ray-operator
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm install kuberay kuberay/kuberay-operator --namespace $namespace    
}

create_ray_cluster() {
    # Check if the namespace already exists and create it if it doesn't
    namespace_exists

    if helm list -n $namespace | grep -q ray-cluster; then
        echo "Ray cluster already exists. Upgrading."
        upgrade_ray_cluster
        return
    else
        echo "Creating Ray cluster."
        helm install ray-cluster kuberay/ray-cluster -f $ray_cluster_value_path --namespace $namespace

        # Get the ray cluster status
        kubectl get raycluster -n $namespace
    fi
}

upgrade_ray_cluster() {
    helm upgrade ray-cluster kuberay/ray-cluster -f $ray_cluster_value_path --namespace $namespace
}

################################################################################
# Call the function with the global gpu_nodes array
################################################################################
label_gpu_nodes "${gpu_nodes[@]}"

################################################################################
# Call the function to install the NVIDIA device plugin for k8s
################################################################################
install_nvidia_device_plugin

################################################################################
# Call the function to create the ray operator in the specified namespace
################################################################################
create_kube_ray_operator

################################################################################
# Call the function to create the ray cluster in the specified namespace
################################################################################
create_ray_cluster
