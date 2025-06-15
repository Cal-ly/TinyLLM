using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Abstractions;
public enum OptimizerTypeEnum
{
    SGD,
    Adam,
    AdamW
}

public enum ActivationTypeEnum
{
    ReLU,
    GELU,
    Tanh,
    Sigmoid
}

public enum WeightInitializationEnum
{
    Xavier,
    Kaiming,
    Normal,
    Uniform
}

public enum ScheduleTypeEnum
{
    Constant,
    Linear,
    Cosine,
    Exponential,
    StepDecay
}
