"""
WebXR Controller Tracking Backend Server
High-performance WebSocket server for receiving and distributing controller poses
"""

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass, asdict
from typing import Dict, Set, Optional, Any
from pathlib import Path

import websockets
from websockets.server import WebSocketServerProtocol
from aiohttp import web

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ControllerPose:
    """Represents a 6DOF controller pose"""
    position: list[float]  # [x, y, z]
    orientation: list[float]  # [x, y, z, w] quaternion
    buttons: dict
    timestamp: float


@dataclass
class PoseFrame:
    """Complete pose data for one or more controllers"""
    timestamp: float
    coordinate_system: str
    controllers: Dict[str, ControllerPose]
    received_at: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'coordinate_system': self.coordinate_system,
            'controllers': {
                hand: {
                    'position': pose.position,
                    'orientation': pose.orientation,
                    'buttons': pose.buttons,
                    'timestamp': pose.timestamp
                }
                for hand, pose in self.controllers.items()
            },
            'received_at': self.received_at
        }


class ControllerTrackingServer:
    """Main server handling WebSocket connections and data distribution"""

    def __init__(self, host: str = '0.0.0.0', ws_port: int = 8000, http_port: int = 8000):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port

        # Connected clients
        self.quest_clients: Set[WebSocketServerProtocol] = set()
        self.visualizer_clients: Set[WebSocketServerProtocol] = set()

        # Latest pose data
        self.latest_pose: Optional[PoseFrame] = None
        self.pose_lock = asyncio.Lock()

        # Statistics
        self.total_frames_received = 0
        self.start_time = time.time()
        self.last_frame_time = 0
        self.frame_rate = 0

        # HTTP app
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Setup HTTP REST API routes"""
        # API routes
        self.app.router.add_get('/api/status', self.handle_status)
        self.app.router.add_get('/api/pose/latest', self.handle_latest_pose)
        self.app.router.add_get('/api/pose/left', self.handle_left_pose)
        self.app.router.add_get('/api/pose/right', self.handle_right_pose)

        # WebSocket route
        self.app.router.add_get('/ws', self.handle_websocket)

        # Frontend routes - serve static files
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/index.html', self.handle_index)
        self.app.router.add_get('/app.js', self.handle_app_js)
        self.app.router.add_get('/styles.css', self.handle_styles_css)

    async def handle_status(self, request: web.Request) -> web.Response:
        """GET /api/status - Server status"""
        uptime = time.time() - self.start_time

        status = {
            'status': 'running',
            'uptime_seconds': uptime,
            'quest_clients': len(self.quest_clients),
            'visualizer_clients': len(self.visualizer_clients),
            'total_frames_received': self.total_frames_received,
            'current_frame_rate': self.frame_rate,
            'has_pose_data': self.latest_pose is not None,
            'last_update': self.last_frame_time if self.last_frame_time else None
        }

        return web.json_response(status)

    async def handle_latest_pose(self, request: web.Request) -> web.Response:
        """GET /api/pose/latest - Latest pose data"""
        async with self.pose_lock:
            if self.latest_pose is None:
                return web.json_response({'error': 'No pose data available'}, status=404)

            return web.json_response(self.latest_pose.to_dict())

    async def handle_left_pose(self, request: web.Request) -> web.Response:
        """GET /api/pose/left - Left controller pose"""
        async with self.pose_lock:
            if self.latest_pose is None or 'left' not in self.latest_pose.controllers:
                return web.json_response({'error': 'No left controller data'}, status=404)

            controller = self.latest_pose.controllers['left']
            return web.json_response({
                'position': controller.position,
                'orientation': controller.orientation,
                'buttons': controller.buttons,
                'timestamp': controller.timestamp
            })

    async def handle_right_pose(self, request: web.Request) -> web.Response:
        """GET /api/pose/right - Right controller pose"""
        async with self.pose_lock:
            if self.latest_pose is None or 'right' not in self.latest_pose.controllers:
                return web.json_response({'error': 'No right controller data'}, status=404)

            controller = self.latest_pose.controllers['right']
            return web.json_response({
                'position': controller.position,
                'orientation': controller.orientation,
                'buttons': controller.buttons,
                'timestamp': controller.timestamp
            })

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve index.html"""
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        index_file = frontend_dir / 'index.html'
        return web.FileResponse(index_file)

    async def handle_app_js(self, request: web.Request) -> web.Response:
        """Serve app.js"""
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        js_file = frontend_dir / 'app.js'
        return web.FileResponse(js_file, headers={'Content-Type': 'application/javascript'})

    async def handle_styles_css(self, request: web.Request) -> web.Response:
        """Serve styles.css"""
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        css_file = frontend_dir / 'styles.css'
        return web.FileResponse(css_file, headers={'Content-Type': 'text/css'})

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint - serves both Quest and visualizer clients"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_type = None

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)

                        # Determine client type from first message
                        if client_type is None:
                            if data.get('type') == 'handshake':
                                if data.get('client') == 'quest3':
                                    client_type = 'quest'
                                    self.quest_clients.add(ws)
                                    logger.info('Quest client connected')
                                    await ws.send_json({'type': 'ack', 'message': 'Quest client registered'})
                                elif data.get('client') == 'visualizer':
                                    client_type = 'visualizer'
                                    self.visualizer_clients.add(ws)
                                    logger.info('Visualizer client connected')
                                    await ws.send_json({'type': 'ack', 'message': 'Visualizer client registered'})

                                    # Send latest pose if available
                                    async with self.pose_lock:
                                        if self.latest_pose:
                                            await ws.send_json(self.latest_pose.to_dict())

                        # Handle pose data from Quest
                        elif client_type == 'quest' and data.get('type') == 'pose':
                            await self.handle_pose_data(data)

                    except json.JSONDecodeError:
                        logger.error('Invalid JSON received')

                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')

        finally:
            # Cleanup on disconnect
            if client_type == 'quest' and ws in self.quest_clients:
                self.quest_clients.remove(ws)
                logger.info('Quest client disconnected')
            elif client_type == 'visualizer' and ws in self.visualizer_clients:
                self.visualizer_clients.remove(ws)
                logger.info('Visualizer client disconnected')

        return ws

    async def handle_pose_data(self, data: dict):
        """Process incoming pose data from Quest"""
        try:
            # Parse controllers
            controllers = {}
            for hand, pose_data in data.get('controllers', {}).items():
                controllers[hand] = ControllerPose(
                    position=pose_data['position'],
                    orientation=pose_data['orientation'],
                    buttons=pose_data.get('buttons', {}),
                    timestamp=data['timestamp']
                )

            # Create pose frame
            pose_frame = PoseFrame(
                timestamp=data['timestamp'],
                coordinate_system=data.get('coordinate_system', 'local'),
                controllers=controllers,
                received_at=time.time()
            )

            # Update latest pose
            async with self.pose_lock:
                self.latest_pose = pose_frame

            # Update statistics
            self.total_frames_received += 1
            current_time = time.time()
            if self.last_frame_time > 0:
                frame_interval = current_time - self.last_frame_time
                if frame_interval > 0:
                    self.frame_rate = round(1.0 / frame_interval, 1)
            self.last_frame_time = current_time

            # Broadcast to visualizer clients
            await self.broadcast_to_visualizers(pose_frame.to_dict())

            # Log periodically
            if self.total_frames_received % 100 == 0:
                logger.info(f'Received {self.total_frames_received} frames, current rate: {self.frame_rate} Hz')

        except Exception as e:
            logger.error(f'Error processing pose data: {e}')

    async def broadcast_to_visualizers(self, data: dict):
        """Broadcast pose data to all connected visualizers"""
        if not self.visualizer_clients:
            return

        # Send to all visualizer clients
        disconnected = set()
        for client in self.visualizer_clients:
            try:
                await client.send_json(data)
            except Exception as e:
                logger.error(f'Error sending to visualizer: {e}')
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            self.visualizer_clients.discard(client)

    def create_ssl_context(self):
        """Create SSL context if certificates exist"""
        cert_dir = Path(__file__).parent / 'certs'
        cert_file = cert_dir / 'cert.pem'
        key_file = cert_dir / 'key.pem'

        if cert_file.exists() and key_file.exists():
            try:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(str(cert_file), str(key_file))
                logger.info(f'✓ SSL certificates found, enabling HTTPS')
                return ssl_context
            except Exception as e:
                logger.warning(f'Failed to load SSL certificates: {e}')
                logger.warning(f'Falling back to HTTP')
                return None
        else:
            logger.info(f'No SSL certificates found at {cert_dir}')
            logger.info(f'Run "python generate_cert.py" to enable HTTPS')
            return None

    async def start_http_server(self):
        """Start the HTTP/WebSocket server"""
        runner = web.AppRunner(self.app)
        await runner.setup()

        # Try to create SSL context
        ssl_context = self.create_ssl_context()

        # Start server with or without SSL
        site = web.TCPSite(runner, self.host, self.http_port, ssl_context=ssl_context)
        await site.start()

        # Log appropriate URL
        protocol = 'https' if ssl_context else 'http'
        logger.info(f'{protocol.upper()} server started on {protocol}://{self.host}:{self.http_port}')

        if ssl_context:
            # Get local IP address for Quest access
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "<your-ip>"
            logger.info(f'✓ WebXR will work! Access from Quest: {protocol}://{local_ip}:{self.http_port}')
        else:
            logger.warning(f'⚠ WebXR requires HTTPS! Generate certificates with: python generate_cert.py')

    async def run(self):
        """Run the server"""
        logger.info('Starting Controller Tracking Server...')
        await self.start_http_server()

        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info('Shutting down server...')


async def main():
    """Main entry point"""
    server = ControllerTrackingServer(
        host='0.0.0.0',
        http_port=8000
    )
    await server.run()


if __name__ == '__main__':
    asyncio.run(main())
