import datetime
import logging
import os
import subprocess
from time import time

import pandas as pd
from clearml import Task
from clearml.backend_api.session.client import APIClient
from omegaconf import OmegaConf


def get_model_path(task_name: str):
    """
    Returns the url to latest model generated during the training in ClearML task(experiment).

    Args:
    ----
        task_name (str): name of the ClearML task

    Raises:
    ------
        ValueError: if the provided task name is not in ClearML

    Returns:
    -------
        str: URL to the model
    """
    task = Task.get_task(task_name=task_name, allow_archived=False)
    if task is None:
        raise ValueError(f"There is no task {task_name} in ClearML")
    models = task.get_models()
    model = models["output"][-1]
    path = model.url
    return path


def get_metric_from_clearml_task(project_name, task_name, metric):
    """
    Get the last metric value from a task.

    Args:
    ----
        project_name (str): requested clearml project
        task_name (str): name of the ClearML task
        metric (str): requested metric

    Raises:
    ------
        ValueError: if the provided task name is not in ClearML

    Returns:
    -------
        float: last metric value
    """
    task = Task.get_task(project_name=project_name, task_name=task_name)
    if task is None:
        raise ValueError(f"There is no task {task_name} in ClearML")
    metrics = task.get_last_scalar_metrics()
    return metrics["Summary"][metric]["last"]


def get_folder_of_checkpoints(task_name):
    """
    Returns the path to the folder with checkpoints of ClearML task.

    Args:
    ----
        task_name (_type_):name of the ClearML task

    Returns:
    -------
        str: path to the folder with checkpoints
    """
    path = get_model_path(task_name)
    # trim the prefix and model name, leave only path to checkpoints folder
    end_index = path.rfind("epoch=")  # Find the last occurrence of the substring
    path = path[:end_index]
    path = path.replace("file://", "")
    return path


def get_root_folder(task: Task, root_path: str = "task.default_root_dir"):
    """
    Retrieves  root folder on CCC of the ClearML experiment (task) from the Configuration Object of the experiment.

    Args:
    ----
        task (Task): Task object of the experiment
        root_path (str, optional): path in configuration to the root folder. Defaults to "task.default_root_dir".

    Returns:
    -------
        Str : path to the root folder of experiment (None if not available)
    """
    configurations = task.get_configuration_objects()
    if "OmegaConf" in configurations.keys():
        string_cfg = configurations["OmegaConf"]
        conf = OmegaConf.create(string_cfg)
        return OmegaConf.select(conf, root_path)
    else:
        return None


def get_task_folder_owner(task: Task, root_path: str = "task.default_root_dir"):
    """
    Returns the owner of the root folder on CCC of the ClearML experiment (task).

    Args:
    ----
        task (Task): Task object of the experiment
        root_path (str, optional): path in configuration to the root folder. Defaults to "task.default_root_dir".

    Returns:
    -------
        Str: user name of root folder owner or "missing folder" if folder is not available or configuration if missing, None in case of error.
    """
    try:
        folder = get_root_folder(task, root_path)
        if folder is None or not os.path.exists(folder):
            return "missing folder"
        result = subprocess.run(
            ["stat", "-c", "%U", folder], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()  # Strip newline from the output
    except subprocess.CalledProcessError:
        return None  # Unable to get owner


def get_ccc_user_name():
    """
    Returns the user name of the logged in CCC user.

    Returns
    -------
        Str: user name of the logged in CCC user
    """
    try:
        result = subprocess.run(["whoami"], capture_output=True, text=True, check=True)
        return result.stdout.strip()  # Strip newline from the output
    except subprocess.CalledProcessError:
        return None  # Unable to get current user


def get_all_clearml_projects() -> pd.DataFrame:
    """
    Extracts a list of all ClearML projects.

    Returns
    -------
        pd.DataFrame: returns the dataframe with the ClearML project details
    """
    all_projects = Task.get_projects()
    all_projects_list = []
    for p in all_projects:
        project = {}
        project["project_name"] = p.name
        project["project_id"] = p.id
        project["user"] = p.user
        project["tags"] = p.tags
        project["system_tags"] = p.system_tags
        all_projects_list.append(project)

    return pd.DataFrame(all_projects_list)


def get_user_clearml_tasks(
    user_id: str = None, archived=True, last_update_in_days=0
) -> pd.DataFrame:
    """
    Retrieves user's tasks from clearML based on a list of conditions.

    Args:
    ----
        user_id (str, optional): user_id of the user in ClearML. Defaults to None.
        archived (bool, optional): To retrieve only archived tasks (True), all otherwise. Defaults to True.
        last_update_in_days (int, optional): number of days after the tasks' last update . Defaults to 0. To retrieve only tasks that were not updated for this number of days

    Returns:
    -------
        pd.DataFrame: dataframe with the tasks details
    """
    if user_id is None:
        logging.error("Could not retrieve tasks for user_id = None")
        return pd.DataFrame()

    projects = get_all_clearml_projects()
    timestamp = time() - 60 * 60 * 24 * last_update_in_days
    system_tags = []
    if archived:
        system_tags = ["archived"]

    try:
        client = APIClient()
        tasks = client.tasks.get_all(
            system_tags=system_tags,
            status_changed=[f"<{datetime.date.fromtimestamp(timestamp)}"],
            user=[user_id],
        )

        all_tasks = []
        for task in tasks:
            flattened_task = {}
            flattened_task["task_id"] = task.id
            flattened_task["task_name"] = task.name
            flattened_task["project_id"] = task.project
            flattened_task["project_name"] = projects[
                projects["project_id"] == task.project
            ]["project_name"].values[0]
            all_tasks.append(flattened_task)

        return pd.DataFrame(all_tasks)

    except Exception as ex:
        logging.warning(
            "Could not retrieve users tasks, {}".format(
                ex.message if hasattr(ex, "message") else ex
            )
        )
        return pd.DataFrame()


def delete_clearml_tasks_by_project(
    tasks: pd.DataFrame, project_name: str, root_path: str = "task.default_root_dir"
):
    """
    Performs deletion of the ClearML tasks including all artifacts in ClearML and CCC.
       It includes following steps:
         - Tasks for deletion are selected  based on the name of the provided project.
         - 'Published' tasks are skipped for deletion
         - Before the deletion the owner of the CCC folder of the current experiment is verified to be the current user.

    Args:
    ----
        tasks (pd.DataFrame): All users tasks assigned for deletion (for all tasks)
        project_name (str): name of the project to delete from the tests
        root_path (str, optional): path in config to the default root folder . Defaults to "task.default_root_dir".

    Error and Warning raised during the delete operation are coming from missing artifacts on CCC
    """
    task_ids = tasks[tasks["project_name"] == project_name]["task_id"].values
    current_user = get_ccc_user_name()

    counter = 0
    for tid in task_ids:
        task = Task.get_task(task_id=tid)
        status = task.get_status()
        print(task.name)
        if status == "published":
            print(f"The task {task.name} was 'Published' and will not be deleted")
            continue
        try:  # deletes from ClearML and files from CCC storage
            folder_owner = get_task_folder_owner(task, root_path)
            if folder_owner == "missing folder" or current_user == folder_owner:
                task.delete(
                    delete_artifacts_and_models=True,
                    skip_models_used_by_other_tasks=True,
                    raise_on_error=False,
                )
                counter = counter + 1

        except Exception as ex:
            logging.warning(
                "Could not delete Task ID={}, {}".format(
                    task.id, ex.message if hasattr(ex, "message") else ex
                )
            )
    print("deleted " + str(counter) + " tasks")


def delete_empty_clearml_projects(user_id: str = None):
    """
    Deletes empty projects that were created by user with provided user_id.

    Args:
    ----
        user_id (str, optional): user_id of ClearML user. Defaults to None.
    """
    if user_id is None:
        logging.error("Could not retrieve tasks for user_id = None")

    projects = get_all_clearml_projects()
    user_projects = projects[projects["user"] == user_id]
    counter = 0
    try:
        client = APIClient()
        for _, p in user_projects.iterrows():
            project_data = client.projects.validate_delete(project=p["project_id"])
            if project_data.tasks == 0 and project_data.models == 0:
                print(f"project {p['project_name']} is empty and will be deleted")
                client.projects.delete(project=p["project_id"])
                counter = counter + 1
    except Exception as ex:
        logging.warning(
            "Could not delete empty projects, {}".format(
                ex.message if hasattr(ex, "message") else ex
            )
        )
    print("deleted " + str(counter) + " empty projects")


def collect_evaluation_baseline_finetune_run_results(
    project_root, task_name, train_sizes
):
    """
    Extract evalaution results from a set of experiments runs in clearmML.
    Expects a set of project of path "project_root"/"sample_size_{train_size}, each containing a 'task_name' experiment
    with "baseline_accuracy" and "fine-tuning_accuracy" metrics.
    Tailored for the format of evaluation eperiments created by /bmfm-targets/run/run_evaluate_finetune_baseline.py.
    todo: make this a bit more generalized to support other project/matric formats.
    """
    df_results = pd.DataFrame(columns=["baseline", "finetuned"])
    res = []
    for size in train_sizes:
        project_name = f"{project_root}/samples_size_{size}"
        baseline_scalar = get_metric_from_clearml_task(
            project_name, task_name, "baseline_accuracy"
        )
        fine_tuning_scalar = get_metric_from_clearml_task(
            project_name, task_name, "fine-tuning_accuracy"
        )
        task_results = {"baseline": baseline_scalar, "finetuned": fine_tuning_scalar}
        res.append(task_results)
    df_results = pd.DataFrame(res)
    df_results.index = train_sizes
    return df_results
