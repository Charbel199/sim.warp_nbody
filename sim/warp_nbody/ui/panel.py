import omni.ui as ui

from ..spawner import (
    spawn_galaxy_disk,
    spawn_sphere,
    spawn_solar_system,
    spawn_random,
    spawn_binary_galaxy,
    spawn_black_hole,
)

PRESETS = ["Galaxy Disk", "Sphere", "Solar System", "Random", "Binary Galaxy", "Black Hole"]

SPAWN_FNS = {
    "Galaxy Disk":   lambda n, G, spread, body_mass: spawn_galaxy_disk(n, radius=50.0, central_mass=1e6, body_mass=body_mass, G=G, spread=spread),
    "Sphere":        lambda n, G, spread, body_mass: spawn_sphere(n, radius=50.0, body_mass=body_mass, speed_scale=0.5, spread=spread),
    "Solar System":  lambda n, G, spread, body_mass: spawn_solar_system(n, G=G, spread=spread),
    "Random":        lambda n, G, spread, body_mass: spawn_random(n, extent=100.0, body_mass=body_mass, speed_scale=1.0, spread=spread),
    "Binary Galaxy": lambda n, G, spread, body_mass: spawn_binary_galaxy(n, G=G, body_mass=body_mass, spread=spread),
    "Black Hole":    lambda n, G, spread, body_mass: spawn_black_hole(n, G=G, body_mass=body_mass, spread=spread),
}


class NBodyPanel:

    def __init__(self):
        self._window          = None
        self._selected_preset = PRESETS[0]
        self._stats_labels    = {}

        self._n_model         = ui.SimpleIntModel(5000)
        self._G_model         = ui.SimpleFloatModel(0.001)
        self._eps_model       = ui.SimpleFloatModel(0.05)
        self._dt_model        = ui.SimpleFloatModel(0.01)
        self._spread_model    = ui.SimpleFloatModel(1.0)
        self._body_mass_model = ui.SimpleFloatModel(1.0)
        self._accretion_model = ui.SimpleBoolModel(True)

    def build(self, on_spawn, on_stop) -> None:
        self._window = ui.Window("N-Body Gravity Simulator", width=340, height=520)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                self._build_preset_row()
                self._build_int_slider("Body Count",     self._n_model,        100,   50000)
                self._build_float_slider("Spread",       self._spread_model,   0.1,   5.0)
                self._build_float_slider("Body Mass",    self._body_mass_model, 0.1,  100.0)
                self._build_float_slider("Gravity G",    self._G_model,        1e-5,  0.1)
                self._build_float_slider("Softening ε",  self._eps_model,      0.01,  1.0)
                self._build_float_slider("Time Step Δt", self._dt_model,       0.001, 0.05)
                self._build_accretion_toggle()
                self._build_action_buttons(on_spawn, on_stop)
                self._build_stats_panel()
                self._build_neural_section()

    def _build_preset_row(self) -> None:
        ui.Label("Preset", height=14)
        with ui.HStack(spacing=4, height=28):
            for preset in PRESETS:
                ui.Button(preset, clicked_fn=lambda p=preset: self._select_preset(p), height=28)

    def _select_preset(self, preset: str) -> None:
        self._selected_preset = preset

    def _build_float_slider(self, label, model, min_val, max_val) -> None:
        with ui.VStack(spacing=2, height=0):
            ui.Label(label, height=14)
            with ui.HStack(spacing=8, height=20):
                ui.FloatSlider(model=model, min=min_val, max=max_val)
                ui.FloatField(model=model, width=64)

    def _build_int_slider(self, label, model, min_val, max_val) -> None:
        with ui.VStack(spacing=2, height=0):
            ui.Label(label, height=14)
            with ui.HStack(spacing=8, height=20):
                ui.IntSlider(model=model, min=min_val, max=max_val)
                ui.IntField(model=model, width=64)

    def _build_accretion_toggle(self) -> None:
        with ui.HStack(height=24, spacing=8):
            ui.Label("Enable Accretion", width=ui.Fraction(1))
            ui.CheckBox(model=self._accretion_model)

    def _build_action_buttons(self, on_spawn, on_stop) -> None:
        with ui.HStack(spacing=8, height=32):
            ui.Button(
                "SPAWN",
                clicked_fn=lambda: on_spawn(
                    preset     = self._selected_preset,
                    n          = self._n_model.get_value_as_int(),
                    G          = self._G_model.get_value_as_float(),
                    softening  = self._eps_model.get_value_as_float(),
                    dt         = self._dt_model.get_value_as_float(),
                    spread     = self._spread_model.get_value_as_float(),
                    body_mass  = self._body_mass_model.get_value_as_float(),
                    accretion  = self._accretion_model.get_value_as_bool(),
                ),
                height=32,
            )
            ui.Button("STOP", clicked_fn=on_stop, height=32)

    def _build_stats_panel(self) -> None:
        ui.Separator()
        with ui.VStack(spacing=2, height=0):
            for key, label in [
                ("active", "Classical Active"),
                ("merges", "Classical Merges"),
                ("neural_active", "Neural Active"),
                ("neural_merges", "Neural Merges"),
                ("pos_error_stat", "Position Error (L2)"),
                ("sim_time", "Sim Time"),
            ]:
                with ui.HStack(height=16):
                    ui.Label(label, width=ui.Fraction(1))
                    lbl = ui.Label("—", alignment=ui.Alignment.RIGHT_CENTER)
                    self._stats_labels[key] = lbl

    def update_stats(self, active: int, merges: int, sim_time: float,
                     neural_active: int = 0, neural_merges: int = 0, pos_error: float = 0.0) -> None:
        self._stats_labels["active"].text         = str(active)
        self._stats_labels["merges"].text         = str(merges)
        self._stats_labels["neural_active"].text  = str(neural_active) if neural_active > 0 else "—"
        self._stats_labels["neural_merges"].text  = str(neural_merges) if neural_active > 0 else "—"
        if neural_active == 0:
            self._stats_labels["pos_error_stat"].text = "—"
        elif pos_error < 0:
            self._stats_labels["pos_error_stat"].text = "N/A (accretion)"
        else:
            self._stats_labels["pos_error_stat"].text = f"{pos_error:.4f}"
        self._stats_labels["sim_time"].text       = f"{sim_time:.1f} s"

    def _build_neural_section(self) -> None:
        self._neural_enabled_model = ui.SimpleBoolModel(False)
        import pathlib
        _default_ckpt = str(pathlib.Path(__file__).resolve().parents[3] / "checkpoints" / "model_best.pt")
        self._checkpoint_path_model = ui.SimpleStringModel(_default_ckpt)
        self._neural_cutoff_model = ui.SimpleFloatModel(2.0)
        self._neural_interval_model = ui.SimpleIntModel(10)
        self._neural_labels = {}

        with ui.CollapsableFrame("Neural Force Field", collapsed=True):
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=24, spacing=8):
                    ui.Label("Enable Neural Mode", width=ui.Fraction(1))
                    ui.CheckBox(model=self._neural_enabled_model)

                with ui.VStack(spacing=2, height=0):
                    ui.Label("Checkpoint Path", height=14)
                    ui.StringField(model=self._checkpoint_path_model, height=20)

                self._build_float_slider("Cutoff Radius", self._neural_cutoff_model, 0.5, 10.0)
                self._build_int_slider("Inference Interval (frames)", self._neural_interval_model, 1, 50)

                ui.Button("Load Model", height=28, clicked_fn=self._on_load_neural_model)

                ui.Separator()
                for key, label in [
                    ("classical_ms", "Classical: — ms/step"),
                    ("neural_ms", "Neural: — ms/step"),
                    ("pos_error", "Position Error: —"),
                ]:
                    lbl = ui.Label(label, height=16)
                    self._neural_labels[key] = lbl

                ui.Separator()
                ui.Button("Generate Training Data", height=28, clicked_fn=self._on_generate_data)
                ui.Button("Train Model", height=28, clicked_fn=self._on_train_model)

    def _on_load_neural_model(self) -> None:
        path = self._checkpoint_path_model.get_value_as_string()
        try:
            from ..neural import NeuralForceField
            self._neural_ff = NeuralForceField(path)
            loaded = self._neural_ff.is_loaded()
            self._neural_labels["neural_ms"].text = f"Model loaded: {loaded}"
        except Exception as e:
            self._neural_labels["neural_ms"].text = f"Load failed: {e}"

    def _on_train_model(self) -> None:
        import threading
        import argparse
        import pathlib

        data_path = str(pathlib.Path(__file__).resolve().parents[3] / "data" / "nbody_dataset.h5")
        checkpoint_dir = str(pathlib.Path(__file__).resolve().parents[3] / "checkpoints")

        def _run():
            try:
                from ..neural.train import train
                self._neural_labels["pos_error"].text = "Training..."
                args = argparse.Namespace(
                    epochs=100,
                    batch_size=32,
                    radius=5.0,
                    data=data_path,
                    output=checkpoint_dir,
                )
                train(args)
                self._neural_labels["pos_error"].text = "Training complete"
                # Auto-update checkpoint path to the trained model
                best_path = str(pathlib.Path(checkpoint_dir) / "model_best.pt")
                self._checkpoint_path_model.set_value(best_path)
            except Exception as e:
                self._neural_labels["pos_error"].text = f"Train error: {e}"

        threading.Thread(target=_run, daemon=True).start()

    def _on_generate_data(self) -> None:
        import threading

        def _run():
            try:
                from ..neural.data_gen import generate_dataset, DataGenConfig
                self._neural_labels["pos_error"].text = "Generating data..."
                generate_dataset(DataGenConfig())
                self._neural_labels["pos_error"].text = "Data generation complete"
            except Exception as e:
                self._neural_labels["pos_error"].text = f"Data gen error: {e}"

        threading.Thread(target=_run, daemon=True).start()

    def update_neural_stats(self, classical_ms: float, neural_ms: float, pos_error: float) -> None:
        if "classical_ms" in self._neural_labels:
            self._neural_labels["classical_ms"].text = f"Classical: {classical_ms:.2f} ms/step"
            self._neural_labels["neural_ms"].text = f"Neural: {neural_ms:.2f} ms/step"
            self._neural_labels["pos_error"].text = f"Position Error: {pos_error:.4f} (mean L2)"

    def get_neural_enabled(self) -> bool:
        return self._neural_enabled_model.get_value_as_bool()

    def get_checkpoint_path(self) -> str:
        return self._checkpoint_path_model.get_value_as_string()

    def get_neural_cutoff(self) -> float:
        return self._neural_cutoff_model.get_value_as_float()

    def get_neural_interval(self) -> int:
        return self._neural_interval_model.get_value_as_int()

    def destroy(self) -> None:
        if self._window:
            self._window.destroy()
            self._window = None
