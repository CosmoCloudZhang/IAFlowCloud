# IAFlow scripts

These modules only parse command-line options, load YAML configuration, and
call the package implementation. Run them after `pip install -e .`:

```bash
python -m IAFlow.Scripts.PrepareData --config Config/NLA/AutoEncoderConv1D.yml
python -m IAFlow.Scripts.TrainAutoEncoder --config Config/NLA/AutoEncoderConv1D.yml
```
