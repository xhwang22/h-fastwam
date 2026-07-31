import os

if os.environ.get("FASTWAM_ROBOTWIN_CPU_RENDER") == "1":
    import sapien

    _set_camera_shader_dir = sapien.render.set_camera_shader_dir

    def _set_cpu_camera_shader_dir(shader_dir: str) -> None:
        _set_camera_shader_dir("default" if shader_dir == "rt" else shader_dir)

    sapien.render.set_camera_shader_dir = _set_cpu_camera_shader_dir
