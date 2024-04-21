# Find the container ID by name
container_id=$(docker ps | grep hatedemo | awk '{print $1}')

# Check if the container is running
if [ -n "$container_id" ]; then
    # Stop the container
    docker stop "$container_id"
    echo "Container stopped."
else
    echo "Container is not running."
fi
