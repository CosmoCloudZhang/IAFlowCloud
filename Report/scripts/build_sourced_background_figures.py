#!/usr/bin/env python3
"""Build report-ready PDFs from credited, published source assets."""

import base64
from pathlib import Path

import cairosvg
from PIL import Image


REPORT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPORT_DIR / "figures"
SOURCE_DIR = FIGURE_DIR / "sources"


def data_uri(path: Path, *, raster_width: int | None = None) -> str:
    if path.suffix.lower() == ".svg":
        content = cairosvg.svg2png(
            url=str(path),
            output_width=raster_width or 1800,
        )
        media_type = "image/png"
    else:
        content = path.read_bytes()
        media_type = "image/jpeg"
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def svg_image(
    source: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    return (
        f'<image href="{source}" x="{x}" y="{y}" '
        f'width="{width}" height="{height}" '
        'preserveAspectRatio="xMidYMid meet"/>'
    )


def write_svg_pdf(svg: str, output_name: str) -> None:
    cairosvg.svg2pdf(
        bytestring=svg.encode("utf-8"),
        write_to=str(FIGURE_DIR / output_name),
    )


def build_planck_figure() -> None:
    source_path = SOURCE_DIR / "planck_cmb_2018.jpg"
    output_path = FIGURE_DIR / "background_planck_cosmology.pdf"
    with Image.open(source_path) as source:
        width = 2400
        height = round(source.height * width / source.width)
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(
            output_path,
            "PDF",
            resolution=300,
            quality=90,
        )


def build_desi_figure() -> None:
    bao = SOURCE_DIR / "desi_dr2_figure8_left.svg"
    dark_energy = SOURCE_DIR / "desi_dr2_figure11.svg"
    bao_source = data_uri(bao, raster_width=1800)
    dark_energy_source = data_uri(dark_energy, raster_width=1800)
    svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="510"
     viewBox="0 0 1040 510">
  <rect width="1040" height="510" fill="white"/>
  <text x="260" y="28" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="24" fill="#111">
    (a) ΛCDM constraints
  </text>
  <text x="780" y="28" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="24" fill="#111">
    (b) Evolving-dark-energy constraints
  </text>
  {svg_image(bao_source, 10, 36, 500, 420)}
  {svg_image(dark_energy_source, 530, 36, 500, 420)}
  <text x="520" y="495" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#333">
    Source: DESI Collaboration, DR2 Results II, Figures 8 (left) and 11
    (CC BY 4.0)
  </text>
</svg>
"""
    write_svg_pdf(svg, "background_desi_dr2.pdf")


def build_survey_figure() -> None:
    comparison = SOURCE_DIR / "kids_legacy_figure14_omega_m_s8.svg"
    source = data_uri(comparison, raster_width=1800)
    svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg" width="620" height="680"
     viewBox="0 0 620 680">
  <rect width="620" height="680" fill="white"/>
  <text x="310" y="30" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="24" fill="#111">
    Published Stage III cosmic-shear comparison
  </text>
  {svg_image(source, 20, 42, 580, 580)}
  <text x="310" y="655" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#333">
    Source: Wright et al. (2025), Figure 14 (CC BY 4.0)
  </text>
</svg>
"""
    write_svg_pdf(svg, "background_s8_surveys.pdf")


def build_lensing_diagram() -> None:
    svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720"
     viewBox="0 0 1200 720">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
            orient="auto-start-reverse" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#17324d"/>
    </marker>
  </defs>
  <rect width="1200" height="720" fill="white"/>
  <style>
    .title { font: bold 25px Arial, sans-serif; fill: #17324d; }
    .label { font: 21px Arial, sans-serif; fill: #18212b; }
    .small { font: 18px Arial, sans-serif; fill: #3c4854; }
    .panel { fill: #f5f8fb; stroke: #b8c6d1; stroke-width: 2; }
  </style>

  <text x="24" y="34" class="title">1. Light paths through intervening matter</text>
  <text x="1175" y="32" text-anchor="end" class="small">AI-generated — GPT-5.6 Sol</text>
  <rect x="20" y="50" width="1160" height="190" rx="14" class="panel"/>
  <ellipse cx="115" cy="140" rx="44" ry="29" fill="#6b8fc4"
           transform="rotate(-18 115 140)"/>
  <path d="M525 120 C545 84 578 88 590 112 C611 82 651 95 648 127
           C684 126 688 168 657 177 C628 199 550 190 526 167
           C500 157 500 130 525 120Z" fill="#d98b45" opacity="0.72"/>
  <circle cx="1082" cy="140" r="28" fill="none" stroke="#17324d" stroke-width="4"/>
  <circle cx="1082" cy="140" r="7" fill="#17324d"/>
  <path d="M159 124 C380 105 470 104 530 119 C680 151 830 166 1053 137"
        fill="none" stroke="#2878b5" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M159 158 C380 176 470 176 530 161 C680 130 830 116 1053 143"
        fill="none" stroke="#2878b5" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M160 141 L1050 141" fill="none" stroke="#8c99a5" stroke-width="2"
        stroke-dasharray="10 8"/>
  <text x="62" y="207" class="label">source galaxy</text>
  <text x="500" y="207" class="label">intervening matter</text>
  <text x="1040" y="207" class="label">observer</text>
  <text x="785" y="85" class="small">deflections exaggerated</text>

  <text x="24" y="282" class="title">2. Convergence and shear change an image</text>
  <rect x="20" y="298" width="1160" height="175" rx="14" class="panel"/>
  <g stroke="#2878b5" stroke-width="5" fill="#dcecf7">
    <circle cx="170" cy="375" r="42"/>
    <circle cx="450" cy="375" r="58"/>
    <ellipse cx="735" cy="375" rx="72" ry="34"/>
    <ellipse cx="1015" cy="375" rx="72" ry="34"
             transform="rotate(45 1015 375)"/>
  </g>
  <g fill="none" stroke="#7c8791" stroke-width="2" stroke-dasharray="8 6">
    <circle cx="450" cy="375" r="42"/>
    <circle cx="735" cy="375" r="42"/>
    <circle cx="1015" cy="375" r="42"/>
  </g>
  <text x="170" y="447" text-anchor="middle" class="label">unlensed</text>
  <text x="450" y="447" text-anchor="middle" class="label">kappa: larger image</text>
  <text x="735" y="447" text-anchor="middle" class="label">gamma1: horizontal shear</text>
  <text x="1015" y="447" text-anchor="middle" class="label">gamma2: rotated shear</text>

  <text x="24" y="516" class="title">3. Spatial scale and wavenumber</text>
  <rect x="20" y="532" width="1160" height="165" rx="14" class="panel"/>
  <text x="600" y="555" text-anchor="middle" class="small">
    Schematic density fluctuations over equal distance intervals
  </text>
  <path d="M55 620 C118 565 181 675 244 620 C307 565 370 675 433 620"
        fill="none" stroke="#3a9d8f" stroke-width="5"/>
  <path d="M760 620 C779 570 798 670 817 620 C836 570 855 670 874 620
           C893 570 912 670 931 620 C950 570 969 670 988 620
           C1007 570 1026 670 1045 620 C1064 570 1083 670 1102 620
           C1121 570 1140 670 1159 620"
        fill="none" stroke="#e07a3f" stroke-width="5"/>
  <path d="M65 671 H190" stroke="#17324d" stroke-width="3"
        marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <path d="M770 671 H840" stroke="#17324d" stroke-width="3"
        marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <text x="244" y="584" text-anchor="middle" class="label">large scale → small k</text>
  <text x="960" y="584" text-anchor="middle" class="label">small scale → large k</text>
  <text x="128" y="693" text-anchor="middle" class="small">long spatial wavelength</text>
  <text x="805" y="693" text-anchor="middle" class="small">short spatial wavelength</text>
  <rect x="460" y="574" width="280" height="100" rx="10" fill="white"
        stroke="#17324d" stroke-width="2"/>
  <text x="600" y="606" text-anchor="middle" class="small">k = 2 pi / spatial wavelength</text>
  <text x="600" y="637" text-anchor="middle" class="label">P_delta(k,z)</text>
  <text x="600" y="662" text-anchor="middle"
        style="font: 15px Arial, sans-serif; fill: #3c4854;">
    strength of density fluctuations at each scale and time
  </text>
</svg>
"""
    write_svg_pdf(svg, "background_lensing_density.pdf")


def build_intrinsic_alignment_diagram() -> None:
    svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720"
     viewBox="0 0 1200 720">
  <defs>
    <marker id="blue-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
            orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#2878b5"/>
    </marker>
    <marker id="orange-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
            orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#d97936"/>
    </marker>
  </defs>
  <rect width="1200" height="720" fill="white"/>
  <style>
    .title { font: bold 28px Arial, sans-serif; fill: #17324d; }
    .label { font: 21px Arial, sans-serif; fill: #18212b; }
    .small { font: 17px Arial, sans-serif; fill: #46515c; }
    .panel { fill: #f6f8fa; stroke: #b7c4ce; stroke-width: 2; }
    .ray { fill: none; stroke: #2878b5; stroke-width: 3; }
    .tide { fill: none; stroke: #d97936; stroke-width: 3; }
  </style>
  <text x="1180" y="18" text-anchor="end" class="small">AI-generated — GPT-5.6 Sol</text>

  <rect x="35" y="24" width="1130" height="92" rx="16" fill="#edf4f8"
        stroke="#8eabbc" stroke-width="2"/>
  <ellipse cx="100" cy="62" rx="43" ry="22" fill="#6b8fc4"
           transform="rotate(-25 100 62)"/>
  <text x="100" y="103" text-anchor="middle" class="small">intrinsic</text>
  <path d="M160 62 H230" class="ray" marker-end="url(#blue-arrow)"/>
  <text x="195" y="48" text-anchor="middle" class="small">shear</text>
  <ellipse cx="300" cy="62" rx="52" ry="20" fill="#4b87b8"
           transform="rotate(-5 300 62)"/>
  <text x="300" y="103" text-anchor="middle" class="small">observed</text>
  <text x="770" y="56" text-anchor="middle"
        style="font: bold 23px Arial, sans-serif; fill: #17324d;">
    measured ellipticity ≈ intrinsic ellipticity +
  </text>
  <text x="770" y="84" text-anchor="middle"
        style="font: bold 23px Arial, sans-serif; fill: #17324d;">
    gravitational shear + measurement noise
  </text>
  <text x="770" y="106" text-anchor="middle" class="small">
    weak-distortion illustration; not literal addition of images
  </text>

  <g>
    <rect x="20" y="145" width="370" height="455" rx="16" class="panel"/>
    <text x="205" y="184" text-anchor="middle" class="title">GG — two lensed images</text>
    <ellipse cx="125" cy="245" rx="47" ry="24" fill="#6b8fc4"
             transform="rotate(-18 125 245)"/>
    <ellipse cx="285" cy="245" rx="47" ry="24" fill="#6b8fc4"
             transform="rotate(22 285 245)"/>
    <path d="M145 270 C180 330 170 410 200 525" class="ray"
          marker-end="url(#blue-arrow)"/>
    <path d="M265 270 C230 330 240 410 210 525" class="ray"
          marker-end="url(#blue-arrow)"/>
    <path d="M145 362 C165 330 195 340 200 360 C218 335 250 350 244 377
             C220 395 166 390 145 362Z" fill="#d97936" opacity="0.65"/>
    <circle cx="205" cy="548" r="20" fill="none" stroke="#17324d" stroke-width="3"/>
    <circle cx="205" cy="548" r="5" fill="#17324d"/>
    <text x="205" y="585" text-anchor="middle" class="small">observer</text>
  </g>

  <g>
    <rect x="415" y="145" width="370" height="455" rx="16" class="panel"/>
    <text x="600" y="184" text-anchor="middle" class="title">II — intrinsic alignment</text>
    <ellipse cx="520" cy="315" rx="58" ry="27" fill="#6b8fc4"
             transform="rotate(28 520 315)"/>
    <ellipse cx="680" cy="315" rx="58" ry="27" fill="#6b8fc4"
             transform="rotate(28 680 315)"/>
    <ellipse cx="600" cy="315" rx="150" ry="100" fill="none"
             stroke="#d97936" stroke-width="3" stroke-dasharray="10 7"/>
    <path d="M600 205 V265" class="tide" marker-end="url(#orange-arrow)"/>
    <path d="M600 425 V365" class="tide" marker-end="url(#orange-arrow)"/>
    <path d="M440 315 H485" class="tide" marker-end="url(#orange-arrow)"/>
    <path d="M760 315 H715" class="tide" marker-end="url(#orange-arrow)"/>
    <text x="600" y="470" text-anchor="middle" class="label">shared tidal environment</text>
    <text x="600" y="508" text-anchor="middle" class="small">
      related orientations before lensing
    </text>
    <circle cx="600" cy="548" r="20" fill="none" stroke="#17324d" stroke-width="3"/>
    <circle cx="600" cy="548" r="5" fill="#17324d"/>
    <text x="600" y="585" text-anchor="middle" class="small">observer</text>
  </g>

  <g>
    <rect x="810" y="145" width="370" height="455" rx="16" class="panel"/>
    <text x="995" y="184" text-anchor="middle" class="title">GI — intrinsic × lensing</text>
    <ellipse cx="1040" cy="235" rx="48" ry="24" fill="#6b8fc4"
             transform="rotate(-25 1040 235)"/>
    <text x="1095" y="240" class="label">G</text>
    <ellipse cx="900" cy="390" rx="55" ry="25" fill="#6b8fc4"
             transform="rotate(25 900 390)"/>
    <text x="830" y="397" class="label">I</text>
    <path d="M1020 260 C990 320 970 410 995 525" class="ray"
          marker-end="url(#blue-arrow)"/>
    <path d="M935 350 C960 320 1000 330 1004 363 C1025 345 1050 365 1038 390
             C1005 407 950 397 935 350Z" fill="#d97936" opacity="0.65"/>
    <path d="M930 355 C915 368 912 374 915 382" class="tide"
          marker-end="url(#orange-arrow)"/>
    <circle cx="995" cy="548" r="20" fill="none" stroke="#17324d" stroke-width="3"/>
    <circle cx="995" cy="548" r="5" fill="#17324d"/>
    <text x="995" y="585" text-anchor="middle" class="small">observer</text>
  </g>

  <text x="390" y="640" text-anchor="end" class="title">full decomposition:</text>
  <text x="430" y="640" class="title">GG</text>
  <text x="495" y="640" class="title">+</text>
  <text x="535" y="640" class="title">GI</text>
  <text x="600" y="640" class="title">+</text>
  <text x="640" y="640" class="title">IG</text>
  <text x="705" y="640" class="title">+</text>
  <text x="745" y="640" class="title">II</text>
  <path d="M525 652 V662 H685 V652" fill="none" stroke="#17324d" stroke-width="2"/>
  <text x="605" y="684" text-anchor="middle" class="small">
    grouped as “GI” when both cross-term orderings apply
  </text>
  <text x="1180" y="713" text-anchor="end" class="small">
    schematic only; panel sizes do not show measured amplitudes
  </text>
</svg>
"""
    write_svg_pdf(svg, "background_intrinsic_alignment.pdf")


def build_model_tradeoff_diagram() -> None:
    svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="760"
     viewBox="0 0 1400 760">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
            orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#17324d"/>
    </marker>
  </defs>
  <rect width="1400" height="760" fill="white"/>
  <style>
    .title { font: bold 25px Arial, sans-serif; fill: #17324d; }
    .label { font: 20px Arial, sans-serif; fill: #18212b; }
    .small { font: 17px Arial, sans-serif; fill: #46515c; }
    .box { stroke: #7892a5; stroke-width: 2; }
    .arrow { fill: none; stroke: #17324d; stroke-width: 3;
             marker-end: url(#arrow); }
  </style>

  <rect x="235" y="20" width="930" height="65" rx="15" fill="#e9f1f6" class="box"/>
  <text x="700" y="60" text-anchor="middle" class="title">
    Galaxy alignments are uncertain across spatial scale and time
  </text>

  <path d="M700 85 V112 H355 V135" class="arrow"/>
  <path d="M700 112 H1045 V135" class="arrow"/>
  <rect x="80" y="135" width="550" height="210" rx="16" fill="#edf6f2" class="box"/>
  <rect x="770" y="135" width="550" height="210" rx="16" fill="#fdf2e9" class="box"/>
  <text x="355" y="170" text-anchor="middle" class="title">Restricted response — NLA example</text>
  <text x="355" y="210" text-anchor="middle" class="label">• fewer adjustable parameters</text>
  <text x="355" y="240" text-anchor="middle" class="label">• easier to constrain and explore</text>
  <text x="355" y="270" text-anchor="middle" class="label">• may miss relevant behavior</text>
  <text x="355" y="300" text-anchor="middle" class="label">• can shift inferred cosmology if inadequate</text>
  <text x="355" y="330" text-anchor="middle" class="small">can be adequate for some data and scales</text>
  <text x="1045" y="170" text-anchor="middle" class="title">More flexible response — TATT example</text>
  <text x="1045" y="210" text-anchor="middle" class="label">• additional tidal-response terms</text>
  <text x="1045" y="240" text-anchor="middle" class="label">• wider family of possible responses</text>
  <text x="1045" y="270" text-anchor="middle" class="label">• more weakly constrained directions</text>
  <text x="1045" y="300" text-anchor="middle" class="label">• extra computation and parameter degeneracies</text>
  <text x="1045" y="330" text-anchor="middle" class="small">more flexibility is not automatically better</text>

  <path d="M355 345 V360 H480 V375" class="arrow"/>
  <path d="M1045 345 V360 H920 V375" class="arrow"/>
  <rect x="125" y="380" width="1150" height="62" rx="14" fill="#eef0fa" class="box"/>
  <text x="700" y="419" text-anchor="middle"
        style="font: bold 23px Arial, sans-serif; fill: #17324d;">
    Can a flexible response family be described with fewer numbers at useful accuracy?
  </text>

  <text x="45" y="480" class="title">Experiment performed in this report</text>
  <text x="1350" y="480" text-anchor="end" class="small">
    NLA-inspired synthetic family
  </text>
  <g>
    <rect x="35" y="510" width="170" height="70" rx="12" fill="#e9f1f6" class="box"/>
    <rect x="245" y="510" width="185" height="70" rx="12" fill="#e9f1f6" class="box"/>
    <rect x="470" y="510" width="160" height="70" rx="12" fill="#edf6f2" class="box"/>
    <rect x="670" y="510" width="160" height="70" rx="12" fill="#edf6f2" class="box"/>
    <rect x="870" y="510" width="190" height="70" rx="12" fill="#fdf2e9" class="box"/>
    <rect x="1100" y="510" width="250" height="70" rx="12" fill="#fdf2e9" class="box"/>
    <text x="120" y="540" text-anchor="middle" class="label">13 sampled</text>
    <text x="120" y="566" text-anchor="middle" class="label">parameters</text>
    <text x="337" y="538" text-anchor="middle" class="label">positive response</text>
    <text x="337" y="563" text-anchor="middle" class="small">AΘ(k,z) on 31 × 101 grid</text>
    <text x="550" y="540" text-anchor="middle" class="label">PCA or</text>
    <text x="550" y="566" text-anchor="middle" class="label">autoencoder</text>
    <text x="750" y="540" text-anchor="middle" class="label">two</text>
    <text x="750" y="566" text-anchor="middle" class="label">coordinates</text>
    <text x="965" y="540" text-anchor="middle" class="label">reconstructed</text>
    <text x="965" y="566" text-anchor="middle" class="label">response</text>
    <text x="1225" y="530" text-anchor="middle" class="small">compare errors on</text>
    <text x="1225" y="551" text-anchor="middle" class="small">the same held-out</text>
    <text x="1225" y="572" text-anchor="middle" class="small">validation samples</text>
    <path d="M205 545 H240" class="arrow"/>
    <path d="M430 545 H465" class="arrow"/>
    <path d="M630 545 H665" class="arrow"/>
    <path d="M830 545 H865" class="arrow"/>
    <path d="M1060 545 H1095" class="arrow"/>
  </g>

  <rect x="145" y="635" width="1110" height="92" rx="14" fill="white"
        stroke="#7d8992" stroke-width="3" stroke-dasharray="12 8"/>
  <text x="700" y="674" text-anchor="middle" class="title">Still required before cosmological use</text>
  <text x="700" y="708" text-anchor="middle" class="label">
    errors in lensing predictions • probabilities for new coordinates • tests of unbiased inference
  </text>
  <text x="1350" y="750" text-anchor="end" class="small">
    AI-generated schematic — GPT-5.6 Sol; not a completed cosmological analysis
  </text>
</svg>
"""
    write_svg_pdf(svg, "background_model_tradeoff.pdf")


def main() -> None:
    build_planck_figure()
    build_desi_figure()
    build_survey_figure()
    build_lensing_diagram()
    build_intrinsic_alignment_diagram()
    build_model_tradeoff_diagram()


if __name__ == "__main__":
    main()
