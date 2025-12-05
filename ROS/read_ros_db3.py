import sys
from pathlib import Path
import json

# The required rosbags imports
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def read_all_topics_rosbags(bag_path, message_limit=5):
    """
    Reads and deserializes messages from all topics in a ROS 2 bag file 
    using the rosbags library for proper CDR decoding.

    Args:
        bag_path (str): The path to the directory containing the .db3 and metadata.yaml files.
        message_limit (int): Maximum number of messages to display per topic.
    """
    
    # IMPORTANT: Change 'ROS2_HUMBLE' if your bag was recorded with a different ROS distro.
    ROS_DISTRO_STORE = Stores.ROS2_HUMBLE 

    try:
        # 1. Initialize the type store and reader
        typestore = get_typestore(ROS_DISTRO_STORE)
        
        print(f"Attempting to read bag from: {bag_path} (Using {ROS_DISTRO_STORE.name} types)")

        with Reader(bag_path) as reader:
            print("\n" + "="*80)
            print(f"BAG SUMMARY | Total Topics: {len(reader.connections)}, Total Messages: {reader.message_count}")
            print("="*80)
            
            # 2. Iterate through all topics (connections) in the bag
            for connection in reader.connections:
                topic_name = connection.topic
                msg_type = connection.msgtype
                total_count = connection.msgcount
                
                if total_count == 0:
                    print(f"\n--- Topic: {topic_name} (Type: {msg_type}) has 0 messages. ---")
                    continue

                print(f"\n{'#'*70}\n### Topic: {topic_name} (Type: {msg_type}, Count: {total_count}) ###")
                print(f"#{'='*68}#")
                
                msg_counter = 0
                
                # 3. Iterate over messages for the current topic
                # reader.messages() yields (connection, timestamp, raw_bytes)
                for conn, timestamp, rawdata in reader.messages(connections=[connection]):
                    if msg_counter >= message_limit:
                        break

                    # Deserialize the raw CDR data into a Python object using the type store
                    # This is the critical step for correct decoding.
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    
                    # Convert the deserialized message object to a readable dictionary/string format
                    try:
                        # Attempt to convert namedtuple structure to dict for cleaner JSON output
                        if hasattr(msg, '_asdict'):
                            output = msg._asdict()
                        else:
                            # Fallback for complex objects that don't convert cleanly (e.g., objects with arrays/sequences)
                            output = str(msg)
                    except Exception:
                         output = str(msg)
                    
                    print(f"  [{msg_counter + 1}] Timestamp: {timestamp} ns (Raw Size: {len(rawdata)} bytes)")
                    # Print the output, using default=str to handle nested rosbags objects
                    print(json.dumps(output, indent=4, default=str)) 
                    
                    msg_counter += 1
                
                if total_count > message_limit:
                     print(f"--- Displaying first {message_limit} messages for brevity. ---")
                
    except Exception as e:
        print(f"\n{'='*80}\nAn UNEXPECTED ERROR occurred. This often means the 'rosbags' library is not installed, or the bag directory structure is incorrect for the reader.")
        print(f"Error Details: {e}")
        print(f"\nNote: Ensure the 'rosbags' library is installed (e.g., `pip install rosbags`).")
        print("="*80)

if __name__ == '__main__':
    # Use the current directory ('.') because both the .db3 file and metadata.yaml
    # are in the same location, mimicking a standard ROS bag directory structure.
    BAG_FOLDER_PATH = "../../Collaboration_user_study/Mugunthan/Easy/rosbag2_2025_11_26-18_13_26" 
    read_all_topics_rosbags(BAG_FOLDER_PATH)