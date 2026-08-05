import asyncio
import websockets
import json

async def test():
    uri = "ws://127.0.0.1:8000/ws/solve"
    file_path = r"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\1000rpm_0p1gL_2cm2\1p0k 30s 0p1g 2cm2_-1V1V_10mVs.csv"
    with open(file_path, "r") as f:
        content = f.read()

    async with websockets.connect(uri) as websocket:
        payload = {
            "config": {
                "scan_rate": 0.010,
                "film_thickness": 1e-4,
                "v_min": -1.0,
                "v_max": 1.0,
                "skip_factor": 10,
                "num_peaks": 4,
                "num_terms": 25,
                "max_iter": 50,
                "tol_ftol": 1e-8,
                "tol_gtol": 1e-7,
                "loss_weight_const": 1.0,
                "pot_col": 8,
                "cur_col": 9
            },
            "file_content": content
        }
        await websocket.send(json.dumps(payload))
        
        while True:
            msg_str = await websocket.recv()
            msg = json.loads(msg_str)
            msg_type = msg.get("type")
            if msg_type == "init":
                print(f"[WS] Init: experimental points = {len(msg['exp_potential'])}")
            elif msg_type == "iter":
                if msg["iter"] % 10 == 0 or msg["iter"] == 1:
                    print(f"[WS] Iteration {msg['iter']}: loss = {msg['loss']:.6f}")
            elif msg_type == "done":
                print("[WS] COMPLETED!")
                print("Parameters:", msg["data"]["parameters"])
                break
            elif msg_type == "error":
                print("[WS] ERROR:", msg["message"])
                break

asyncio.run(test())
