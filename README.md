### Presentation for the PES MRes Mini Project 1: All-Optical Modulation of Microring Resonators for Photonic Neural Networks

This package contains the Python scripts used to generate the presentation slides with Manim [Manim](https://www.manim.community/)
 and [Manim-slides](https://github.com/jeertmans/manim-slides).

### Installation of Poetry
The installation instructions for poetry can be found on their website. Don't forget to add the new installation directory into your PATH.

Installing the package using poetry
To install all dependencies for this repo simply run `poetry install`. Some basic usage and commands for poetry can be found [here](https://python-poetry.org/docs/cli/) or by running `poetry help`.

With poetry, commands are executed inside the virtual environment and to signify this each command should be pre-pended with `poetry run`. For example to run pytest in the package you should run:

```poetry run pytest```

After poetry run you can specify any commands and arguments you need.

You can also run poetry shell which opens a poetry environment within the terminal. You can also integrate this into the Python interpreters in VS Code. Start by finding the path of the poetry's Python (for instance, by running which python on Linux). Then, input this path into the "Select Interpreter" section of VS Code settings.

### Main Commands
**Render**

`manim-slides render mp1-presentation/slides.py <Slide Name> --open`

**HTML Conversion**

`manim-slides convert <Slide Name> <File Name>.html --open

**Convert to PDF**

`manim-slides convert --to=pdf <Slide Name> <File Name>.pdf`

**Concatenate Multiple Slides**

`manim-slides [present|convert] Slide1 Slide2`

To concatenate the final presentation:

```manim-slides convert OpeningTitle Contents Motivation MRROperation MicroringTransmissionModulation ModulationTypes KerrEquations AllOpticalModulation ExperimentDiagram FrequencyModulationResponse ApplicationPart1 ApplicationPart2 ApplicationPart3 ApplicationPart7 ThankYou mp1_presentation.html --open```
