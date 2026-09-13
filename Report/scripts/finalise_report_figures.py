#!/usr/bin/env python3
"""
Assemble report figures by editing PDF objects while preserving scientific plots.

Run build_sourced_background_figures.py first to prepare replacement mathematical
labels. Inputs under sources/before_section2_finalisation
are immutable copies of the supplied PDFs. Attribution remains in the captions
and sources/README.md. No contours or power-spectrum curves are recomputed.
"""

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

FIGURES = Path(__file__).resolve().parents[1] / "figures"
ORIGINALS = FIGURES / "sources" / "before_section2_finalisation"
SCHEMATICS = FIGURES / "sources" / "schematics"


def remove_credit_text(
    page,
):
    """
    Delete the single added Source credit text object from a supplied panel.
    
    Arguments:
        page (pypdf.PageObject):
            Page whose separately drawn credit is to be removed.
    """
    stream = page.get_contents()
    operations = stream.operations
    blocks = []
    start = None
    
    for index, (operands, operator) in enumerate(operations):
        if operator == b"BT":
            start = index
        elif operator == b"ET" and start is not None:
            block = operations[start:index + 1]
            if any("Source:" in str(values) for values, _ in block):
                blocks.append((start, index + 1))
            start = None
    
    if len(blocks) != 1:
        raise ValueError(f"Expected one credit text object, found {len(blocks)}.")
    
    first, last = blocks[0]
    stream.operations = operations[:first] + operations[last:]
    page.replace_contents(stream)
    
    if "Source:" in page.extract_text():
        raise ValueError("Credit text remains after removing its text object.")


def set_crop(
    page,
    bounds,
):
    """
    Trim only whitespace surrounding a cleaned panel.
    
    Arguments:
        page (pypdf.PageObject):
            Page with unwanted text already deleted.
        bounds (tuple):
            Left, bottom, right, top page coordinates in points.
    """
    page.mediabox = RectangleObject(bounds)
    page.cropbox = RectangleObject(bounds)
    page.trimbox = RectangleObject(bounds)


def write_page(
    page,
    path,
):
    """
    Write one edited page and retain its embedded resources.
    
    Arguments:
        page (pypdf.PageObject):
            Completed figure page.
        path (pathlib.Path):
            Destination PDF.
    """
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(path)


def build_desi():
    """
    Remove the added DESI footer without modifying either published panel.
    """
    source = PdfWriter(clone_from=ORIGINALS / "background_desi_dr2.pdf")
    page = source.pages[0]
    remove_credit_text(page)
    set_crop(page, (0, 33.5, 780, 382.5))
    write_page(page, FIGURES / "background_desi_dr2.pdf")


def build_surveys():
    """
    Retain only the published survey comparison, without the added credit footer.
    """
    source = PdfWriter(clone_from=ORIGINALS / "background_s8_surveys.pdf")
    historical = source.pages[0]
    remove_credit_text(historical)
    set_crop(historical, (0, 35, 465, 510))
    write_page(historical, FIGURES / "background_s8_surveys.pdf")


def replace_power_labels():
    """
    Replace two axis labels and the matter-power title without changing curves.
    """
    source = PdfWriter(clone_from=ORIGINALS / "ia_power_spectra_panels.pdf")
    page = source.pages[0]
    stream = page.get_contents()
    operations = stream.operations
    blocks = []
    
    for index, (operands, operator) in enumerate(operations):
        if operator != b"cm" or len(operands) != 6:
            continue
        vertical_label = (
            list(operands[:4]) == [0, 1, -1, 0]
            and any(abs(float(operands[4]) - x) < 0.001 for x in (17.2, 561.136))
        )
        matter_title = (
            list(operands[:4]) == [1, 0, 0, 1]
            and abs(float(operands[4]) - 135.8) < 0.001
            and abs(float(operands[5]) - 234.244) < 0.001
        )
        if not (vertical_label or matter_title):
            continue
        if operations[index - 1][1] != b"q":
            raise ValueError("Expected a saved graphics state around the label.")
        depth = 1
        end = index
        
        while depth:
            end += 1
            operator = operations[end][1]
            depth += int(operator == b"q") - int(operator == b"Q")
        
        blocks.append((index - 1, end + 1))
    
    if len(blocks) != 3:
        raise ValueError(f"Expected three label blocks, found {len(blocks)}.")
    
    for first, last in reversed(blocks):
        del operations[first:last]
    
    stream.operations = operations
    page.replace_contents(stream)
    labels = PdfReader(SCHEMATICS / "power_spectrum_labels.pdf").pages[0]
    page.merge_page(labels)
    write_page(page, FIGURES / "ia_power_spectra_panels.pdf")


def main():
    """
    Rebuild the three PDF assemblies from preserved inputs and fresh overlays.
    """
    build_desi()
    build_surveys()
    replace_power_labels()


if __name__ == "__main__":
    main()
