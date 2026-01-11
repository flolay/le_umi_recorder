"""
Rerun URDF Visualizer - Visualize and control robot joints in Rerun.

This module provides a class to load URDF files into Rerun and update
joint positions in real-time.
"""

import re
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np

import rerun as rr
import yourdfpy


class RerunURDFVisualizer:
    """
    A class to visualize URDF robots in Rerun with dynamic joint control.

    Example:
        >>> viz = RerunURDFVisualizer("path/to/robot.urdf")
        >>> viz.init("my_robot")
        >>> viz.set_joint_positions({"shoulder_pan": 0.5, "elbow_flex": -0.3})
    """

    def __init__(self, urdf_path: str | Path):
        """
        Initialize the visualizer with a URDF file.

        Args:
            urdf_path: Path to the URDF file
        """
        self.urdf_path = Path(urdf_path).resolve()
        self.urdf_dir = self.urdf_path.parent

        # Load URDF with yourdfpy for joint information
        self.robot = yourdfpy.URDF.load(
            str(self.urdf_path),
            filename_handler=self._filename_handler,
        )

        # Cache joint information
        self._joint_info = self._extract_joint_info()
        self._entity_paths = self._build_entity_paths()

    def _filename_handler(self, fname: str) -> str:
        """Handle package:// URLs in URDF mesh references."""
        if fname.startswith("package://"):
            # Remove package:// prefix and resolve relative to URDF directory
            relative_path = fname.replace("package://", "")
            return str(self.urdf_dir / relative_path)
        return fname

    def _create_resolved_urdf(self) -> Path:
        """Create a temporary URDF file with resolved mesh paths."""
        urdf_content = self.urdf_path.read_text()

        # Replace package:// URLs with absolute paths
        def replace_package_url(match):
            relative_path = match.group(1)
            absolute_path = str(self.urdf_dir / relative_path)
            return f'filename="{absolute_path}"'

        resolved_content = re.sub(
            r'filename="package://([^"]+)"',
            replace_package_url,
            urdf_content
        )

        # Write to temp file (persistent for Rerun to read)
        self._temp_urdf = tempfile.NamedTemporaryFile(
            mode='w', suffix='.urdf', delete=False
        )
        self._temp_urdf.write(resolved_content)
        self._temp_urdf.flush()
        return Path(self._temp_urdf.name)

    def _extract_joint_info(self) -> dict:
        """Extract joint names, types, axes, and limits from URDF."""
        joints = {}
        for joint_name, joint in self.robot.joint_map.items():
            if joint.type in ("revolute", "continuous", "prismatic"):
                joints[joint_name] = {
                    "type": joint.type,
                    "axis": np.array(joint.axis) if joint.axis is not None else np.array([0, 0, 1]),
                    "parent": joint.parent,
                    "child": joint.child,
                    "origin": joint.origin,
                    "limit": joint.limit,
                }
        return joints

    def _build_entity_paths(self) -> dict:
        """Build Rerun entity paths for each joint based on kinematic chain.

        Rerun's URDF loader creates paths using link names only:
        - Structure: base_link/child_link1/child_link2/...
        - Transform for a joint is applied to its child link's entity
        """
        paths = {}

        # Build link path lookup - maps link name to its full entity path
        link_paths = {}
        root_link = self.robot.base_link
        link_paths[root_link] = root_link

        # Build map of parent link -> list of (joint_name, child_link)
        parent_to_children = {}
        for joint_name, info in self._joint_info.items():
            parent = info["parent"]
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append((joint_name, info["child"]))

        def build_paths(link_name: str):
            """Recursively build paths through kinematic chain."""
            if link_name not in parent_to_children:
                return

            parent_path = link_paths[link_name]
            for joint_name, child_link in parent_to_children[link_name]:
                # Child link path: parent_path/child_link
                child_path = f"{parent_path}/{child_link}"
                link_paths[child_link] = child_path
                # Joint transform is applied at child link entity
                paths[joint_name] = child_path
                build_paths(child_link)

        build_paths(root_link)
        return paths

    def init(self, app_id: str = "urdf_visualizer", spawn: bool = True):
        """
        Initialize Rerun and load the URDF.

        Args:
            app_id: Application ID for Rerun
            spawn: Whether to spawn a new Rerun viewer
        """
        rr.init(app_id, spawn=spawn)
        self._load_urdf_to_rerun()

    def connect(self, addr: str = "127.0.0.1:9876"):
        """Connect to an existing Rerun viewer."""
        rr.connect_tcp(addr)
        self._load_urdf_to_rerun()

    def _load_urdf_to_rerun(self):
        """Load the URDF into Rerun by manually logging meshes."""
        import trimesh

        # Log each link's visual meshes
        for link_name, link in self.robot.link_map.items():
            entity_path = self._get_link_entity_path(link_name)

            for i, visual in enumerate(link.visuals):
                if visual.geometry is None:
                    continue

                mesh = visual.geometry.mesh
                if mesh is None:
                    continue

                # Load the mesh file
                mesh_path = self._filename_handler(mesh.filename)
                try:
                    trimesh_mesh = trimesh.load(mesh_path)
                except Exception as e:
                    print(f"Warning: Could not load mesh {mesh_path}: {e}")
                    continue

                # Handle mesh geometry
                if isinstance(trimesh_mesh, trimesh.Scene):
                    # Flatten scene to single mesh
                    trimesh_mesh = trimesh_mesh.dump(concatenate=True)

                if not hasattr(trimesh_mesh, 'vertices'):
                    continue

                vertices = np.array(trimesh_mesh.vertices, dtype=np.float32)
                faces = np.array(trimesh_mesh.faces, dtype=np.uint32)

                # Apply visual origin transform
                if visual.origin is not None:
                    # Transform vertices by visual origin
                    ones = np.ones((vertices.shape[0], 1))
                    vertices_h = np.hstack([vertices, ones])
                    vertices = (visual.origin @ vertices_h.T).T[:, :3]

                # Get color from material
                color = [200, 200, 200, 255]  # Default gray
                if visual.material and visual.material.color is not None:
                    mat_color = visual.material.color
                    # Handle different color object types
                    if hasattr(mat_color, 'rgba'):
                        rgba = mat_color.rgba
                    elif hasattr(mat_color, '__iter__'):
                        rgba = list(mat_color)
                    else:
                        rgba = [0.8, 0.8, 0.8, 1.0]
                    color = [int(c * 255) for c in rgba[:4]]

                # Log the mesh
                mesh_entity = f"{entity_path}/visual_{i}"
                rr.log(
                    mesh_entity,
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_colors=np.tile(color, (len(vertices), 1)).astype(np.uint8),
                    ),
                    static=True,
                )

        # Log fixed joint transforms (these don't change)
        self._log_fixed_joints()

        # Set initial joint positions to zero
        self.set_joint_positions({name: 0.0 for name in self._joint_info})

    def _log_fixed_joints(self):
        """Log transforms for fixed joints."""
        from scipy.spatial.transform import Rotation

        for joint_name, joint in self.robot.joint_map.items():
            if joint.type != "fixed":
                continue

            # Get entity path for the child link
            child_path = self._get_link_entity_path(joint.child)

            if joint.origin is not None:
                origin_xyz = joint.origin[:3, 3].tolist()
                origin_rot = Rotation.from_matrix(joint.origin[:3, :3])
                quat = origin_rot.as_quat()

                rr.log(child_path, rr.Transform3D(
                    translation=origin_xyz,
                    rotation=rr.Quaternion(xyzw=quat.tolist()),
                ), static=True)

    def _get_link_entity_path(self, link_name: str) -> str:
        """Get the entity path for a link."""
        # Build path by traversing from root to this link
        if link_name == self.robot.base_link:
            return link_name

        # Find the path through the kinematic chain
        for joint_name, info in self._joint_info.items():
            if info["child"] == link_name:
                parent_path = self._get_link_entity_path(info["parent"])
                return f"{parent_path}/{link_name}"

        # Check fixed joints too
        for joint_name, joint in self.robot.joint_map.items():
            if joint.child == link_name:
                parent_path = self._get_link_entity_path(joint.parent)
                return f"{parent_path}/{link_name}"

        return link_name

    @property
    def joint_names(self) -> list[str]:
        """Get list of controllable joint names."""
        return list(self._joint_info.keys())

    def get_joint_limits(self, joint_name: str) -> tuple[float, float]:
        """Get (lower, upper) limits for a joint."""
        info = self._joint_info.get(joint_name)
        if info and info["limit"]:
            return (info["limit"].lower, info["limit"].upper)
        return (-np.pi, np.pi)  # Default for continuous joints

    def set_joint_position(self, joint_name: str, position: float):
        """
        Set a single joint position.

        Args:
            joint_name: Name of the joint
            position: Joint position in radians (revolute) or meters (prismatic)
        """
        self.set_joint_positions({joint_name: position})

    def set_joint_positions(self, positions: dict[str, float]):
        """
        Set multiple joint positions at once.

        Args:
            positions: Dictionary mapping joint names to positions
        """
        from scipy.spatial.transform import Rotation

        for joint_name, position in positions.items():
            if joint_name not in self._joint_info:
                continue

            info = self._joint_info[joint_name]
            entity_path = self._entity_paths.get(joint_name)

            if entity_path is None:
                continue

            axis = info["axis"]
            origin = info["origin"]

            # Get original joint origin transform (4x4 matrix)
            if origin is not None:
                origin_xyz = origin[:3, 3].tolist()
                origin_rot = Rotation.from_matrix(origin[:3, :3])
            else:
                origin_xyz = [0.0, 0.0, 0.0]
                origin_rot = Rotation.identity()

            if info["type"] in ("revolute", "continuous"):
                # Create rotation around joint axis
                joint_rot = Rotation.from_rotvec(axis * position)
                # Compose: origin_rotation * joint_rotation
                combined_rot = origin_rot * joint_rot
                # Convert to quaternion for Rerun (xyzw format)
                quat = combined_rot.as_quat()  # Returns [x, y, z, w]

                rr.log(entity_path, rr.Transform3D(
                    translation=origin_xyz,
                    rotation=rr.Quaternion(xyzw=quat.tolist()),
                ))
            elif info["type"] == "prismatic":
                # Translation along axis added to origin
                delta = axis * position
                total_translation = [origin_xyz[i] + delta[i] for i in range(3)]
                quat = origin_rot.as_quat()
                rr.log(entity_path, rr.Transform3D(
                    translation=total_translation,
                    rotation=rr.Quaternion(xyzw=quat.tolist()),
                ))

    def set_joint_positions_array(self, positions: np.ndarray | list[float]):
        """
        Set joint positions using an array (in order of joint_names).

        Args:
            positions: Array of joint positions
        """
        positions_dict = dict(zip(self.joint_names, positions))
        self.set_joint_positions(positions_dict)

    def animate(
        self,
        trajectory: np.ndarray,
        joint_names: Optional[list[str]] = None,
        fps: float = 30.0,
    ):
        """
        Animate through a trajectory of joint positions.

        Args:
            trajectory: Array of shape (n_steps, n_joints)
            joint_names: Joint names corresponding to columns (default: all joints)
            fps: Frames per second for animation timing
        """
        if joint_names is None:
            joint_names = self.joint_names

        dt = 1.0 / fps

        for i, positions in enumerate(trajectory):
            rr.set_time_seconds("animation", i * dt)
            positions_dict = dict(zip(joint_names, positions))
            self.set_joint_positions(positions_dict)
