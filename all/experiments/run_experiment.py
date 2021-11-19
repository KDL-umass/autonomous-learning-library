from .single_env_experiment import SingleEnvExperiment
from .parallel_env_experiment import ParallelEnvExperiment
from all.presets import ParallelPreset
import torch


def run_experiment(
        agents,
        envs,
        frames,
        logdir='runs',
        quiet=False,
        render=False,
        test_episodes=100,
        write_loss=True,
        writer="tensorboard",
        loadfile=""
):
    if not isinstance(agents, list):
        agents = [agents]

    if not isinstance(envs, list):
        envs = [envs]

    for env in envs:
        for preset_builder in agents:
            env.seed(0)
            preset = preset_builder.env(env).build()
            if loadfile == "":
                preset = preset_builder.env(env).build()
            else:
                preset = torch.load(loadfile)
            make_experiment = get_experiment_type(preset)
            experiment = make_experiment(
                preset,
                env,
                train_steps=frames,
                logdir=logdir,
                quiet=quiet,
                render=render,
                write_loss=write_loss,
                writer=writer,
            )
            print("train")
            print(loadfile)
            print(experiment._writer.log_dir)

            if loadfile != "":
                with open("runs/loaded_dirs.txt", 'a+') as f:
                    f.write(loadfile+", "+experiment._writer.log_dir)
                    f.close()

            experiment.train(frames=frames)
            experiment.save()
            experiment.test(episodes=test_episodes)
            experiment.close()


def get_experiment_type(preset):
    if isinstance(preset, ParallelPreset):
        return ParallelEnvExperiment
    return SingleEnvExperiment
