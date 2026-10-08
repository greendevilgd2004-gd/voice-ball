"""
Step 4: bridge between the voice controller and the game page.

The game (HTML) connects to  ws://localhost:8765  and expects messages like
    {"command": "left", "confidence": 0.92}
This script listens to the microphone (VoiceController) and sends every
recognised command to all connected game pages.

Install:  pip install websockets
Run:      python voice_server.py            (add --device 1 to pick the mic)
Then:     open the game HTML file in the browser (it reconnects by itself).

Files needed in the same folder: features.py, voice_controller.py, model.joblib
"""

import argparse
import asyncio
import json

import websockets

from voice_controller import VoiceController

clients = set()


async def handler(ws):
    """One browser tab = one connection."""
    clients.add(ws)
    print("Game connected. Clients:", len(clients))
    try:
        await ws.send(json.dumps({"type": "status", "state": "listening"}))
        async for _ in ws:      # we ignore anything the page sends
            pass
    finally:
        clients.discard(ws)
        print("Game disconnected. Clients:", len(clients))


async def broadcast(message):
    if clients:
        await asyncio.gather(
            *(c.send(message) for c in list(clients)), return_exceptions=True
        )


async def main(args):
    loop = asyncio.get_running_loop()
    outbox = asyncio.Queue()
    voice = None

    def on_command(label):
        # Called from the controller's own thread -> hand over to asyncio safely.
        conf = float(getattr(voice, "last_confidence", 1.0))
        msg = json.dumps({"command": label, "confidence": round(conf, 2)})
        loop.call_soon_threadsafe(outbox.put_nowait, msg)

    voice = VoiceController(
        device=args.device, onset_rms=args.onset,
        on_command=on_command, verbose=True,
    )
    voice.start()
    try:
        async with websockets.serve(handler, args.host, args.port):
            print(f"Voice server running on ws://{args.host}:{args.port}")
            print("Open the game page now. Ctrl+C to stop.")
            while True:
                msg = await outbox.get()
                print("-> game:", msg)
                await broadcast(msg)
    finally:
        voice.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice -> game WebSocket bridge")
    parser.add_argument("--device", type=int, default=None, help="microphone number")
    parser.add_argument("--onset", type=float, default=800.0, help="word-start sound level")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        print("\nStopped.")
