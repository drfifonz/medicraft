import wandb


class WandbTracker:
    def __init__(self, project_name: str, hyperparameters: dict) -> None:
        # wandb.init(project=project_name, name=experiment_name)
        # self.run = wandb.run
        print("WandbTracker initialized")
        print(f"Project name: {project_name}")

    def log(self, metrics):
        self.run.log(metrics)

    def finish(self):
        self.run.finish()
