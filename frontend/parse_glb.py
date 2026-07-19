import json
import struct

def parse_glb_mesh_details(filename):
    with open(filename, 'rb') as f:
        f.read(12) # header
        chunk_length = struct.unpack('<I', f.read(4))[0]
        f.read(4) # type
        json_data = f.read(chunk_length)
        gltf = json.loads(json_data.decode('utf-8'))
        
        print("Nodes:")
        for node in gltf.get('nodes', []):
            if 'name' in node:
                print(f"Node: {node['name']}")
        
        print("\nMeshes:")
        for mesh in gltf.get('meshes', []):
            if 'name' in mesh:
                print(f"Mesh: {mesh['name']}")
                
        print("\nMaterials:")
        for mat in gltf.get('materials', []):
            if 'name' in mat:
                print(f"Material: {mat['name']}")

if __name__ == "__main__":
    parse_glb_mesh_details("public/Robot.glb")
