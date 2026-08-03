import logging

logger = logging.getLogger(__name__)


class RTSPSimulator:
    """RTSP Stream Simulator for testing multi-camera scaling without physical RTSP hardware.

    Generates simulated RTSP camera endpoint configurations or maps local video files / dummy sources
    for scaling tests (e.g. 4, 16, 50, 100 cameras).
    """

    @staticmethod
    def generate_mock_camera_configs(count: int = 4, base_rtsp_url: str = "rtsp://localhost:8554") -> list[dict[str, str]]:
        """Generate a list of simulated camera configuration dicts.

        Args:
            count: Number of simulated camera streams to generate.
            base_rtsp_url: Base URL pattern for RTSP server.

        Returns:
            List of dicts: [{'camera_id': 'cam_01', 'rtsp_url': 'rtsp://.../cam_01'}, ...]
        """
        configs = []
        for i in range(1, count + 1):
            cam_id = f"cam_{i:02d}"
            url = f"{base_rtsp_url}/{cam_id}"
            configs.append({
                "camera_id": cam_id,
                "rtsp_url": url,
                "location": f"Zone {((i - 1) % 5) + 1} - Area {i}"
            })
        return configs
