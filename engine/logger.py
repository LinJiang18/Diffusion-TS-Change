import os
import time

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
