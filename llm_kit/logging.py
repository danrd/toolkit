import json


def prepare_prompt_artifact(prompts_artifact, task_idx:int, prompt_data, report_types=('json', 'txt')):
    if 'json' in report_types:
        # 1. JSON format (structured data)
        with prompts_artifact.new_file(f"task_{task_idx}_prompt.json") as f:
            json.dump(prompt_data, f, indent=2)

    if 'txt' in report_types:
        # 2. Text format (human-readable, like print output)
        with prompts_artifact.new_file(f"task_{task_idx}_prompt_readable.txt") as f:
            f.write(f"{'='*80}\n")
            f.write(f"Task: {prompt_data['task_id']}\n")
            f.write(f"Description: {prompt_data['run_description']}\n")
            f.write(f"Length: {prompt_data['prompt_length']} characters\n")
            f.write(f"{'='*80}\n")
            f.write(f"Generation result:\n\n{prompt_data['generation_result']}\n\n")
            f.write(f"{'='*80}\n")
            f.write(f"{prompt_data['prompt_text']}\n\n")

    return prompts_artifact

def load_checkpoint_from_wandb(run):
    """Load checkpoint from wandb artifacts if resuming."""
    try:
        # Try to get the latest checkpoint artifact
        checkpoint_artifact = run.use_artifact(f"checkpoint-{run.id}:latest", type="checkpoint")
        checkpoint_dir = checkpoint_artifact.download()

        with open(f"{checkpoint_dir}/checkpoint.json", 'r') as f:
            checkpoint = json.load(f)
            return checkpoint
    except Exception:
        print("No checkpoint found, starting fresh")
        return None

def save_checkpoint_to_wandb(run, tasks_summary, prompts_data, processed_tasks, solved_tasks):
    """Save checkpoint to wandb artifacts."""
    import wandb

    checkpoint_data = {
        'processed_tasks': list(processed_tasks),
        'tasks_summary': tasks_summary.get_dataframe().to_dict(),
        'prompts_data': prompts_data,
        'solved_tasks': solved_tasks,
    }

    # Create checkpoint artifact
    checkpoint_artifact = wandb.Artifact(f"checkpoint-{run.id}", type="checkpoint")

    # Save checkpoint data as JSON
    with checkpoint_artifact.new_file("checkpoint.json") as f:
        json.dump(checkpoint_data, f, indent=2)

    # Log the artifact. wandb keeps every version of an artifact, and a checkpoint holds the whole state each
    # time (over a megabyte once a few hundred tasks are done), so a run that saves after every task fills the
    # account's storage with copies of itself: the earlier versions are deleted once the new one is up.
    logged = run.log_artifact(checkpoint_artifact)
    if logged is not None:
        logged.wait()
        delete_older_checkpoints(run, keep_version=logged.version)
    print(f"Checkpoint saved with {len(processed_tasks)} processed tasks")


def delete_older_checkpoints(run, keep_version):
    """Delete every version of this run's checkpoint artifact but `keep_version`. A failure here (no network, no
    permission) is reported and does not stop the run: the checkpoint it would have pruned is already saved."""
    import wandb

    try:
        name = f"{run.entity}/{run.project}/checkpoint-{run.id}"
        for artifact in list(wandb.Api().artifacts("checkpoint", name)):
            if artifact.version != keep_version:
                artifact.delete(delete_aliases=True)
    except Exception as error:  # noqa: BLE001 - best effort, see above
        print(f"Could not delete earlier checkpoint versions: {type(error).__name__}: {error}")
