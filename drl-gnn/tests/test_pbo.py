import shutil
import subprocess

import json


def test_pbo(tmp_path):
    # copy a pbo script pair (.py/.json)
    pbo_json = str(tmp_path / "sphere_2d.json")
    shutil.copy("external/pbo/envs/sphere_2d.py", str(tmp_path))
    shutil.copy("external/pbo/envs/sphere_2d.json", str(tmp_path))

    with open(pbo_json, "r") as f:
        pbo_params = json.load(f)

    pbo_params["n_gen"] = 2
    pbo_params["n_avg"] = 1
    pbo_params["n_ind"] = 1
    pbo_params["n_cpu"] = 1

    with open(pbo_json, "w") as f:
        json.dump(pbo_params, f)
    assert (tmp_path / "sphere_2d.json").exists()

    # run the pbo script
    result = subprocess.run(
        ["pbo", "sphere_2d.json"], capture_output=True, text=True, cwd=str(tmp_path)
    )

    # check the output
    assert result.returncode == 0
