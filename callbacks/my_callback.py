import matplotlib.pyplot as plt

from warprec.utils.callback import WarpRecCallback


class ComputeNDCGOverIterations(WarpRecCallback):
    def __init__(self, *args, **kwargs):
        self._save_path = kwargs.get("save_path", None)
        self._ndcg_scores = []

    def on_trial_save(self, iteration, trials, trial, **info):
        ndcg_score = trial.last_result.get("score", 0.0)
        self._ndcg_scores.append(ndcg_score)

    def on_training_complete(self, model, *args, **kwargs):
        iterations = list(range(1, len(self._ndcg_scores) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self._ndcg_scores, marker="o", linestyle="-")

        plt.title("nDCG@5 over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("nDCG@5")
        plt.grid(True)
        plt.xticks(iterations)
        plt.tight_layout()

        if self._save_path:
            try:
                plt.savefig(self._save_path)
                print(f"Plot successfully save to: {self._save_path}")
            except Exception as e:
                print(f"Error during the saving process in {self._save_path}: {e}")
            plt.close()
        else:
            plt.show()
