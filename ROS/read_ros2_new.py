import sys
from pathlib import Path

# Use the specific Reader and Type Store imports
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def read_tf_messages_v6(bag_path):
    """
    Reads and deserializes messages from the /tf topic, compatible with older 
    rosbags versions that do not accept 'typestore' in Reader.__init__.

    Args:
        bag_path (str): The path to the directory containing the .db3 and metadata.yaml files.
    """
    # --- Configuration ---
    bag_dir = Path(bag_path)
    TARGET_TOPIC = '/tf'
    
    # IMPORTANT: Change 'ROS2_HUMBLE' to match your ROS distribution.
    ROS_DISTRO_STORE = Stores.ROS2_HUMBLE 

    if not bag_dir.is_dir():
        print(f"Error: Bag directory not found at {bag_dir}")
        sys.exit(1)

    print(f"--- ROS 2 Bag Reader for Topic: {TARGET_TOPIC} (Targeting {ROS_DISTRO_STORE.name}) ---")

    try:
        # Get the type store which contains the definitions needed
        typestore = get_typestore(ROS_DISTRO_STORE)
        
        # Get the internal deserialize function from the typestore
        deserialize_cdr_func = typestore.deserialize_cdr
        
        # --- CRITICAL CORRECTION HERE ---
        # Initialize Reader WITHOUT the 'typestore' argument to avoid the error.
        with Reader(bag_dir) as reader:
        # --- END CRITICAL CORRECTION ---
            
            # 1. Filter connections (topics)
            tf_connections = [
                conn for conn in reader.connections 
                if conn.topic == TARGET_TOPIC
            ]

            if not tf_connections:
                print(f"Error: Topic '{TARGET_TOPIC}' not found in the bag.")
                return
            
            tf_connection = tf_connections[0]
            total_message_count = tf_connection.msgcount
            
            print(f"Found topic '{TARGET_TOPIC}' with type: {tf_connection.msgtype}")
            print(f"Total messages to process: {total_message_count}")
            print("-" * 50)
            
            # 2. Iterate over raw messages
            msg_counter = 0
            # reader.messages() yields (connection, timestamp, rawdata/bytes)
            for connection, timestamp, rawdata in reader.messages(connections=tf_connections):
                
                # Explicitly call the function retrieved from the typestore to unpack the data
                msg = deserialize_cdr_func(rawdata, connection.msgtype)

                msg_counter += 1
                
                # The message object now has the 'transforms' attribute
                print(f"[{timestamp} ns] /tf Message ({msg_counter}/{total_message_count}) contains {len(msg.transforms)} transform(s):")
                
                # 3. Process each transform
                for transform in msg.transforms:
                    header = transform.header
                    t = transform.transform.translation
                    r = transform.transform.rotation
                    
                    print(f"  > Frame: {header.frame_id} -> {transform.child_frame_id}")
                    print(f"    - Time: {header.stamp.sec}.{header.stamp.nanosec} sec (in header)")
                    print(f"    - Translation (Vector): x={t.x:.4f}, y={t.y:.4f}, z={t.z:.4f}")
                    print(f"    - Rotation (Quaternion): x={r.x:.4f}, y={r.y:.4f}, z={r.z:.4f}, w={r.w:.4f}")
                
                # Add a conditional break for testing to avoid huge output
                if msg_counter >= 5:
                    print("\n--- Displaying first 5 messages for brevity. Remove this break to process all messages. ---")
                    break

                print("-" * 50)

    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")

if __name__ == '__main__':
    # REPLACE THIS with the path to your folder 
    BAG_FOLDER_PATH = '../../Collaboration_user_study/Mugunthan/Easy/rosbag2_2025_11_26-18_13_26' 
    
    if BAG_FOLDER_PATH == './your_bag_folder':
        print("Please update the 'BAG_FOLDER_PATH' variable with the actual path to your ROS 2 bag directory.")
        sys.exit(1)

    read_tf_messages_v6(BAG_FOLDER_PATH)