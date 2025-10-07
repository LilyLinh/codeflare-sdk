#!/usr/bin/env python3

"""
E2E Test for Ray GCS Fault Tolerance

Based on Ray documentation: https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/kuberay-gcs-ft.html

Tests the complete GCS fault tolerance workflow:
1. Setup namespace (test-project)
2. Setup Redis infrastructure
3. Deploy RayCluster with GCS fault tolerance enabled
4. Create detached actor
5. Verify data persistence in Redis
6. Kill GCS process in head pod
7. Verify actor persistence after GCS restart
8. Clean up Redis keys
"""

import time
import subprocess
import pytest
import os
import tempfile
from codeflare_sdk import Cluster
from codeflare_sdk.ray.cluster.config import ClusterConfiguration
from support import *
import secrets
import string


class KubernetesHelper:
    """Utility class for Kubernetes operations"""

    @staticmethod
    def run_kubectl(args, check=True, capture_output=False, timeout=60, input=None):
        """Execute kubectl command with timeout and optional input"""
        cmd = ["kubectl"] + args
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            input=input,
        )

    @staticmethod
    def get_pod_names(label_selector, namespace):
        """Get pod names matching label selector"""
        result = KubernetesHelper.run_kubectl(
            [
                "get",
                "pods",
                "-l",
                label_selector,
                "-n",
                namespace,
                "-o",
                "jsonpath={.items[*].metadata.name}",
            ],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()
        return []

    @staticmethod
    def wait_for_condition(
        resource_type, selector, condition, namespace, timeout="60s"
    ):
        """Wait for a condition on a resource"""
        return KubernetesHelper.run_kubectl(
            [
                "wait",
                f"--for=condition={condition}",
                resource_type,
                selector,
                f"--timeout={timeout}",
                "-n",
                namespace,
            ]
        )

    @staticmethod
    def exec_in_pod(pod_name, namespace, command, input=None):
        """Execute command in pod with optional input"""
        cmd = ["exec", pod_name, "-n", namespace, "--"] + command
        return KubernetesHelper.run_kubectl(cmd, capture_output=True, input=input)

    @staticmethod
    def apply_yaml(yaml_content, resource_name):
        """Apply YAML content to cluster"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            result = KubernetesHelper.run_kubectl(
                ["apply", "-f", temp_file], capture_output=True
            )
            if result.returncode != 0:
                print(f"Failed to apply {resource_name}: {result.stderr}")
                return False
            print(f"{resource_name} applied successfully")
            return True
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestGCSFaultTolerance:
    """Test class for GCS fault tolerance workflow"""

    def setup_method(self):
        """Setup test configuration"""
        initialize_kubernetes_client(self)
        self.namespace = "test-project"
        self.ray_cluster_name = "raycluster-external-redis"
        # Use environment variable for Redis password, fallback to generated password
        self.redis_password = os.environ.get(
            "REDIS_PASSWORD", self._generate_secure_password()
        )
        self.cluster = None

    def teardown_method(self):
        """Clean up after test"""
        if hasattr(self, "cluster") and self.cluster:
            try:
                self.cluster.down()
            except:
                pass
        self._cleanup_existing_resources()

    def _generate_secure_password(self):
        """Generate a secure random password"""
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(32))

    def _setup_namespace(self):
        """Setup namespace"""
        print(f"Setting up namespace '{self.namespace}'...")
        create_namespace_with_name(self, self.namespace)

    def _cleanup_existing_resources(self):
        """Clean up any existing resources"""
        print("Cleaning up existing resources...")

        # Delete RayCluster using CodeFlare SDK if cluster object exists
        if hasattr(self, "cluster") and self.cluster:
            try:
                self.cluster.down()
                print("RayCluster deleted using CodeFlare SDK")
            except Exception as e:
                print(f"Error deleting RayCluster with SDK: {e}")
                # Fallback to kubectl
                try:
                    KubernetesHelper.run_kubectl(
                        [
                            "delete",
                            "raycluster",
                            self.ray_cluster_name,
                            "-n",
                            self.namespace,
                            "--ignore-not-found=true",
                            "--timeout=30s",
                        ]
                    )
                except:
                    pass
        else:
            # Delete RayCluster using kubectl
            try:
                KubernetesHelper.run_kubectl(
                    [
                        "delete",
                        "raycluster",
                        self.ray_cluster_name,
                        "-n",
                        self.namespace,
                        "--ignore-not-found=true",
                        "--timeout=30s",
                    ]
                )
            except:
                pass

        # Delete Redis resources
        try:
            KubernetesHelper.run_kubectl(
                [
                    "delete",
                    "deployment",
                    "redis",
                    "-n",
                    self.namespace,
                    "--ignore-not-found=true",
                ]
            )
        except:
            pass

        try:
            KubernetesHelper.run_kubectl(
                [
                    "delete",
                    "service",
                    "redis",
                    "-n",
                    self.namespace,
                    "--ignore-not-found=true",
                ]
            )
        except:
            pass

        try:
            KubernetesHelper.run_kubectl(
                [
                    "delete",
                    "secret",
                    "redis-password-secret",
                    "-n",
                    self.namespace,
                    "--ignore-not-found=true",
                ]
            )
        except:
            pass

        try:
            KubernetesHelper.run_kubectl(
                [
                    "delete",
                    "configmap",
                    "redis-config",
                    "-n",
                    self.namespace,
                    "--ignore-not-found=true",
                ]
            )
        except:
            pass

            time.sleep(5)

    def _setup_redis_infrastructure(self):
        """Setup Redis infrastructure based on Ray documentation"""
        print("Setting up Redis infrastructure...")

        # Create Redis password secret
        import base64

        redis_secret_yaml = f"""apiVersion: v1
kind: Secret
metadata:
  name: redis-password-secret
  namespace: {self.namespace}
type: Opaque
data:
  password: {base64.b64encode(self.redis_password.encode()).decode()}
"""

        # Create Redis service
        redis_service_yaml = f"""apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: {self.namespace}
spec:
  type: ClusterIP
  ports:
  - port: 6379
    targetPort: 6379
    protocol: TCP
  selector:
    app: redis
"""

        # Create Redis deployment with secure password handling
        redis_deployment_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: {self.namespace}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.4.0
        ports:
        - containerPort: 6379
        command: ["sh", "-c", "redis-server --requirepass $(cat /etc/redis/password) --maxmemory 256mb --maxmemory-policy allkeys-lru"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-password-secret
              key: password
        volumeMounts:
        - name: redis-password
          mountPath: /etc/redis/password
          subPath: password
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "256Mi"
            cpu: "200m"
      volumes:
      - name: redis-password
        secret:
          secretName: redis-password-secret
"""

        # Apply Redis resources
        KubernetesHelper.apply_yaml(redis_secret_yaml, "Redis Secret")
        KubernetesHelper.apply_yaml(redis_service_yaml, "Redis Service")
        KubernetesHelper.apply_yaml(redis_deployment_yaml, "Redis Deployment")

        # Wait for Redis to be ready
        print("Waiting for Redis to be ready...")
        try:
            KubernetesHelper.wait_for_condition(
                "pod", "-l app=redis", "Ready", self.namespace, "60s"
            )
            print("Redis is ready")
        except Exception as e:
            pytest.fail(f"Redis failed to become ready: {e}")

    def _deploy_ray_cluster_with_gcs_ft(self):
        """Deploy RayCluster with GCS fault tolerance using CodeFlare SDK"""
        print("Deploying RayCluster with GCS fault tolerance using CodeFlare SDK...")

        # Create cluster configuration with GCS fault tolerance
        config = ClusterConfiguration(
            name=self.ray_cluster_name,
            namespace=self.namespace,
            head_cpu_requests="0.5",
            head_cpu_limits="1",
            head_memory_requests=2,
            head_memory_limits=4,
            worker_cpu_requests="0.3",
            worker_cpu_limits="0.5",
            worker_memory_requests=1,
            worker_memory_limits=2,
            num_workers=1,
            local_queue=None,
            enable_gcs_ft=True,
            redis_address="redis:6379",
            redis_password_secret={"name": "redis-password-secret", "key": "password"},
            external_storage_namespace=self.namespace,
            envs={
                "RAY_gcs_rpc_server_reconnect_timeout_s": "60",
                "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
                "RAY_DISABLE_IMPORT_WARNING": "1",
            },
        )

        print(f"Cluster configuration created:")
        print(f"   - Name: {config.name}")
        print(f"   - Namespace: {config.namespace}")
        print(f"   - GCS Fault Tolerance: {config.enable_gcs_ft}")
        print(f"   - Redis Address: {config.redis_address}")
        print(
            f"   - Head Memory: {config.head_memory_requests}G requests, {config.head_memory_limits}G limits"
        )
        print(
            f"   - Worker Memory: {config.worker_memory_requests}G requests, {config.worker_memory_limits}G limits"
        )
        print(f"   - Workers: {config.num_workers}")

        self.cluster = Cluster(config)
        self.cluster.up()
        print("Cluster deployment initiated")

        print("Waiting for cluster to be ready...")
        try:
            self.cluster.wait_ready(timeout=300)  # 5 minutes timeout
            print("Cluster is ready!")

            # Get cluster status
            status = self.cluster.status()
            print(f"Cluster status: {status}")

        except Exception as e:
            pytest.fail(f"Ray cluster failed to become ready: {e}")

    def _create_detached_actor_script(self):
        """Create the detached actor script"""
        detached_actor_script = """import ray
            import time

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get(self):
        return self.value

# Initialize Ray with a specific namespace
ray.init(address="auto", namespace="gcs_test_namespace")

# Create detached actor
counter = Counter.options(name="detached_counter", lifetime="detached").remote()

# Increment counter
result = ray.get(counter.increment.remote())
print(f"Counter value: {result}")

# Keep the script running to maintain the actor
time.sleep(10)
"""
        return detached_actor_script

    def _create_increment_script(self):
        """Create the increment counter script"""
        increment_script = """import ray

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get(self):
        return self.value

# Initialize Ray with the same namespace
ray.init(address="auto", namespace="gcs_test_namespace")

# Get the detached actor
try:
    counter = ray.get_actor("detached_counter")
    result = ray.get(counter.increment.remote())
    print(f"Counter value after increment: {result}")
        except Exception as e:
    print(f"Error accessing detached actor: {e}")
"""
        return increment_script

    def _copy_script_to_pod(self, pod_name, script_content, script_name):
        """Copy script to pod using exec instead of cp to avoid permission issues"""
        try:
            # Use kubectl exec to create the script directly in the pod
            result = KubernetesHelper.run_kubectl(
                [
                    "exec",
                    pod_name,
                    "-n",
                    self.namespace,
                    "--",
                    "sh",
                    "-c",
                    f"cat > /tmp/{script_name} << 'EOF'\n{script_content}\nEOF",
                ],
                capture_output=True,
            )

            if result.returncode != 0:
                print(f"Failed to create {script_name} in pod: {result.stderr}")
                return False

            print(f"{script_name} created in pod")
            return True
        except Exception as e:
            print(f"Error creating {script_name} in pod: {e}")
            return False

    def _get_head_pod_name(self):
        """Get the head pod name"""
        head_pods = KubernetesHelper.get_pod_names(
            f"ray.io/cluster={self.ray_cluster_name},ray.io/node-type=head",
            self.namespace,
        )
        if not head_pods:
            pytest.fail("No head pod found")
        return head_pods[0]

    def _get_worker_pod_name(self):
        """Get a worker pod name"""
        worker_pods = KubernetesHelper.get_pod_names(
            f"ray.io/cluster={self.ray_cluster_name},ray.io/node-type=worker",
            self.namespace,
        )
        if not worker_pods:
            pytest.fail("No worker pod found")
        return worker_pods[0]

    def _create_detached_actor(self):
        """Create a detached actor"""
        print("Creating detached actor...")

        head_pod = self._get_head_pod_name()
        worker_pod = self._get_worker_pod_name()

        # Copy scripts to both pods
        detached_script = self._create_detached_actor_script()
        increment_script = self._create_increment_script()

        self._copy_script_to_pod(worker_pod, detached_script, "detached_actor.py")
        self._copy_script_to_pod(head_pod, increment_script, "increment_counter.py")

        # Run detached actor script on worker pod
        print("Starting detached actor on worker pod...")
        result = KubernetesHelper.exec_in_pod(
            worker_pod, self.namespace, ["python3", "/tmp/detached_actor.py"]
        )

        if result.returncode != 0:
            print(f"Detached actor script output: {result.stdout}")
            print(f"Detached actor script error: {result.stderr}")

        time.sleep(5)  # Give time for actor to be created

        # Test accessing the actor
        print("Testing detached actor access...")
        result = KubernetesHelper.exec_in_pod(
            head_pod, self.namespace, ["python3", "/tmp/increment_counter.py"]
        )

        if result.returncode == 0:
            print(f"Detached actor test result: {result.stdout.strip()}")
            return True
        else:
            print(f"Failed to access detached actor: {result.stderr}")
            return False

    def _check_redis_data(self):
        """Check data stored in Redis"""
        print("Checking Redis data...")

        # Get Redis pod
        redis_pods = KubernetesHelper.get_pod_names("app=redis", self.namespace)
        if not redis_pods:
            pytest.fail("No Redis pod found")

        redis_pod = redis_pods[0]

        # Check Redis keys using stdin to avoid password in command line
        result = KubernetesHelper.exec_in_pod(
            redis_pod,
            self.namespace,
            ["redis-cli", "--no-auth-warning"],
            input=f"AUTH {self.redis_password}\nKEYS *\nQUIT\n",
        )

        if result.returncode == 0:
            keys = result.stdout.strip()
            print(f"Redis keys found: {keys}")
            return len(keys) > 0
        else:
            print(f"Failed to check Redis keys: {result.stderr}")
            return False

    def _kill_gcs_process(self):
        """Kill the GCS process in the head pod"""
        print("Killing GCS process in head pod...")

        head_pod = self._get_head_pod_name()

        # Kill the GCS process
        result = KubernetesHelper.exec_in_pod(
            head_pod, self.namespace, ["pkill", "-f", "gcs_server"]
        )

        print(f"GCS process killed (exit code: {result.returncode})")

        # Wait for GCS to restart
        print("Waiting for GCS to restart...")
        time.sleep(30)

        # Check if head pod is still running
        head_pods = KubernetesHelper.get_pod_names(
            f"ray.io/cluster={self.ray_cluster_name},ray.io/node-type=head",
            self.namespace,
        )
        if head_pods:
            print("Head pod is still running after GCS restart")
            return True
        else:
            print("Head pod is not running after GCS restart")
            return False

    def _test_actor_persistence(self):
        """Test if the detached actor persists after GCS restart"""
        print("Testing actor persistence after GCS restart...")

        head_pod = self._get_head_pod_name()

        # Try to access the detached actor again
        result = KubernetesHelper.exec_in_pod(
            head_pod, self.namespace, ["python3", "/tmp/increment_counter.py"]
        )

        if result.returncode == 0:
            print(f"Actor persistence test result: {result.stdout.strip()}")
            return True
        else:
            print(f"Failed to access actor after GCS restart: {result.stderr}")
            return False

    def _cleanup_redis_keys(self):
        """Clean up Redis keys"""
        print("Cleaning up Redis keys...")

        # Get Redis pod
        redis_pods = KubernetesHelper.get_pod_names("app=redis", self.namespace)
        if not redis_pods:
            print("No Redis pod found for cleanup")
            return

        redis_pod = redis_pods[0]

        # Check if Redis pod is running
        result = KubernetesHelper.run_kubectl(
            [
                "get",
                "pod",
                redis_pod,
                "-n",
                self.namespace,
                "-o",
                "jsonpath={.status.phase}",
            ],
            capture_output=True,
        )

        if result.returncode != 0 or result.stdout.strip() != "Running":
            print("Redis pod is not running, skipping key cleanup")
            return

        # Delete all keys with better error handling using stdin
        try:
            result = KubernetesHelper.exec_in_pod(
                redis_pod,
                self.namespace,
                ["redis-cli", "--no-auth-warning"],
                input=f"AUTH {self.redis_password}\nFLUSHALL\nQUIT\n",
            )

            if result.returncode == 0:
                print("Redis keys cleaned up")
            else:
                print(f"Failed to clean up Redis keys: {result.stderr}")
        except Exception as e:
            print(f"Error during Redis cleanup: {e}")

    def test_gcs_fault_tolerance_workflow(self):
        """Test the complete GCS fault tolerance workflow"""
        print("\nTesting Complete GCS Fault Tolerance Workflow")
        print("=" * 70)

        # Step 1: Setup namespace
        self._setup_namespace()

        # Step 2: Setup Redis infrastructure
        self._setup_redis_infrastructure()

        # Step 3: Deploy RayCluster with GCS fault tolerance
        self._deploy_ray_cluster_with_gcs_ft()

        # Step 4: Create detached actor
        if not self._create_detached_actor():
            pytest.fail("Failed to create detached actor")

        # Step 5: Check Redis data
        if not self._check_redis_data():
            pytest.fail("No data found in Redis")

        # Step 6: Kill GCS process
        if not self._kill_gcs_process():
            pytest.fail("GCS process kill/restart failed")

        # Step 7: Test actor persistence
        if not self._test_actor_persistence():
            pytest.fail("Actor persistence test failed")

        print("\nGCS Fault Tolerance Test Completed Successfully!")
        print(
            "All steps passed: Redis setup, RayCluster deployment, detached actor creation, GCS restart, and actor persistence"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
