---
layout: default
title: Instructions
parent: Dataset
nav_order: 4
---

# Instructions structure

Here we present a structure of the scene `instructions.json` files.

```json
{
    "info": {
        "version": string,
        "split": string,
        "author": string,
        "date": "date"
    },
    "instructions": [
        {
            "split": string,
            "image_filename": string,
            "image_index": int,
            "instruction": string,
            "program": [
                {
                    "input": null / list(int),
                    "type": string,
                    "input_value": null / string,
                    "output": list(int)
                },
                ...
            ],
            "template_id": string,
            "task_id": string,
            "template_internal_idx": int,
            "template_overall_idx": int,
            "instruction_idx": int
        },
        ...
    ]
}
```

Structure of the program field resembles the structure presented in [CLEVR-IEP](https://arxiv.org/abs/1705.03633). An example program for an instruction *Measure the weight of the blue object.* related to 5 element scene can be seen below:
```json
[
    {
        "input": null,
        "type": "scene",
        "input_value": null,
        "output": [                 <-- all elements in the scene returned
            0,
            1,
            2,
            3,
            4
        ]
    },
    {
        "type": "filter_colour",
        "input": [
            0,
            1,
            2,
            3,
            4
        ],
        "input_value": "blue",      <-- value of the input for filter_colour
        "output": [
            4
        ]
    },
    {
        "type": "query_weight",
        "input": [
            4
        ],
        "input_value": null,
        "output": [
            123.3                   <-- returned ground truth
        ]
    }
]
```