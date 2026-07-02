import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def z_rc(R: float, C: float, freq: np.ndarray) -> np.ndarray:

    omega = 2 * np.pi * freq
    return R / (1 + 1j * omega * R * C)


def z_ecm(params: dict, freq: np.ndarray) -> np.ndarray:

    R0 = params["R0"]
    z = R0 + z_rc(params["R1"], params["C1"], freq) \
           + z_rc(params["R2"], params["C2"], freq) \
           + z_rc(params["R3"], params["C3"], freq)
    return z


def read_eis_csv(filepath: str) -> list[pd.DataFrame]:

    with open(filepath, "r", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()

    header = None
    groups = []
    current: list[str] = []

    for line in raw_lines:
        stripped = line.strip()
        if header is None:
            header = stripped
            continue
        if stripped == "":
            if current:
                groups.append(current)
                current = []
        else:
            current.append(stripped)
    if current:
        groups.append(current)

    result = []
    for g in groups:
        from io import StringIO
        text = header + "\n" + "\n".join(g)
        df = pd.read_csv(StringIO(text))
        df.columns = ["freq", "Zre", "Zim"]
        df = df.astype(float)
        result.append(df)

    print(f"[EIS CSV]  {len(result)}")
    return result


def read_ecm_csv(filepath: str) -> list[dict]:

    df = pd.read_csv(filepath)

    col_names = ["R0", "R1", "C1", "R2", "C2", "R3", "C3"]
    if df.shape[1] >= 7:
        df = df.iloc[:, :7]
        df.columns = col_names
    else:
        raise ValueError(f"ECM CSV Insufficient rows")

    params_list = [row.to_dict() for _, row in df.iterrows()]
    print(f"[ECM CSV]  {len(params_list)}")
    return params_list


def compute_errors(z_meas: np.ndarray, z_fit: np.ndarray):

    res = np.abs(z_meas - z_fit)
    mae = np.mean(res)
    mse = np.mean(res ** 2)
    return mae, mse, res


def sigma_from_ci(residuals: np.ndarray, ci: float) -> float:

    alpha = 1 - ci
    z_score = stats.norm.ppf(1 - alpha / 2)
    sigma = np.std(residuals, ddof=1)
    sigma_bound = z_score * sigma
    return sigma, sigma_bound, z_score


def plot_nyquist(eis_groups: list[pd.DataFrame],
                 ecm_params: list[dict],
                 ci: float = 0.95,
                 output_path: str = "nyquist_fit.png"):

    n_groups = min(len(eis_groups), len(ecm_params))



    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i / max(n_groups, 10)) for i in range(n_groups)]

    fig, ax = plt.subplots(figsize=(9, 7))

    all_results = []

    for i in range(n_groups):
        df = eis_groups[i]
        params = ecm_params[i]
        color = colors[i]

        freq = df["freq"].values
        z_meas = df["Zre"].values + 1j * df["Zim"].values


        z_fit = z_ecm(params, freq)


        mae, mse, residuals = compute_errors(z_meas, z_fit)
        sigma, sigma_bound, z_score = sigma_from_ci(residuals, ci)

        all_results.append({
            "group": i + 1,
            "MAE": mae,
            "MSE": mse,
            "sigma": sigma,
            f"sigma_bound (CI={ci*100:.0f}%)": sigma_bound,
            "z_score": z_score,
        })

        label_meas = f"Case {i+1} Measured"
        label_fit  = f"Case {i+1} ECM Fit"
        label_ci   = f"Case {i+1} {ci*100:.0f}% CI Band"


        ax.scatter(df["Zre"], -df["Zim"],
                   color=color, marker="o", s=40, zorder=5,
                   label=label_meas)


        sort_idx = np.argsort(z_fit.real)
        ax.plot(z_fit.real[sort_idx], -z_fit.imag[sort_idx],
                color=color, linewidth=1.8, linestyle="--",
                label=label_fit)


        ax.fill_between(
            z_fit.real[sort_idx],
            (-z_fit.imag - sigma_bound)[sort_idx],
            (-z_fit.imag + sigma_bound)[sort_idx],
            color=color, alpha=0.15,
            label=label_ci
        )

    ax.set_xlabel(r"$Z_{\rm re}$ / Ω", fontsize=13)
    ax.set_ylabel(r"$-Z_{\rm im}$ / Ω", fontsize=13)
    ax.set_title("Nyquist Plot — R-(RC)-(RC)-(RC) ECM Fitting", fontsize=14)
    ax.legend(fontsize=8, loc="upper left", ncol=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n figure: {output_path}")


    print("\n" + "=" * 60)
    print(f"  fit_report  （CI = {ci*100:.0f}%）")
    print("=" * 60)
    for r in all_results:
        g = r["group"]
        print(f"\n  Load profile {g}:")
        print(f"    MAE           = {r['MAE']:.6e} Ω")
        print(f"    MSE           = {r['MSE']:.6e} Ω²")
        print(f"    σ = {r['sigma']:.6e} Ω")
        key = f"sigma_bound (CI={ci*100:.0f}%)"
        print(f"    {ci*100:.0f}% CI  = {r[key]:.6e} Ω  "
              f"（z={r['z_score']:.4f}·σ）")
    print("=" * 60)

    return all_results



def main():
    parser = argparse.ArgumentParser(
        description="EIS Nyquist"
    )
    parser.add_argument("--eis_csv", default=r"E:\data\eis_data_path.csv")
    parser.add_argument("--ecm_csv", default=r"E:\data\ecm_data_path.csv")
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--output", default=r"E:\data")
    args = parser.parse_args()


    eis_groups = read_eis_csv(args.eis_csv)
    ecm_params = read_ecm_csv(args.ecm_csv)



    plot_nyquist(eis_groups, ecm_params, ci=args.ci, output_path=args.output)


if __name__ == "__main__":
    main()