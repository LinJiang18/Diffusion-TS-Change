import os
import time
import torch
import numpy as np

class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        log_path = os.path.join(save_dir, "log.txt")
        self.log_file = open(log_path, "a")

    def log_info(self, msg):
        """
        Log text information with timestamp.
        """
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{time_str}] {msg}"
        print(line)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def add_scalar(self, tag, scalar_value, global_step):
        """
        Log scalar values (e.g., loss) to text log.
        """
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{time_str}] {tag} | step={global_step} | value={scalar_value:.6f}"
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def save_config(self, config):
        import yaml
        path = os.path.join(self.save_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(config, f)

    def close(self):
        self.log_file.close()


class DecompTrace:
    def __init__(self, dtype="float16"):
        self.dtype = dtype
        self.data = {}  # t -> {"trend_cum": np.ndarray, "season_cum": np.ndarray}

    @torch.no_grad()
    def log(self, t, img, trend_cum, season_cum):
        t_int = int(t.flatten()[0].item()) if torch.is_tensor(t) else int(t)

        img = img.detach().cpu().numpy()
        tc = trend_cum.detach().cpu().numpy()
        sc = season_cum.detach().cpu().numpy()

        if self.dtype is not None:
            img = img.astype(self.dtype, copy=False)
            tc = tc.astype(self.dtype, copy=False)
            sc = sc.astype(self.dtype, copy=False)

        self.data[t_int] = {"img": img, "trend_cum": tc, "season_cum": sc}



def save_trace_npz(trace, path):
    ts = sorted(trace.data.keys())
    save_dict = {"t": np.array(ts, dtype=np.int32)}

    for t in ts:
        save_dict[f"img/{t}"] = trace.data[t]["img"]
        save_dict[f"trend_cum/{t}"] = trace.data[t]["trend_cum"]
        save_dict[f"season_cum/{t}"] = trace.data[t]["season_cum"]

    np.savez_compressed(path, **save_dict)


def load_trace_npz(path):
    z = np.load(path, allow_pickle=False)
    ts = list(z["t"].astype(int))
    data = {}
    for t in ts:
        data[t] = {
            "img": z[f"img/{t}"],
            "trend_cum": z[f"trend_cum/{t}"],
            "season_cum": z[f"season_cum/{t}"],
        }
    return data
