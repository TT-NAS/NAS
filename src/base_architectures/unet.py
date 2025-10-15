unet_paper = (
    [  # layers: [convs+deconvs, convs+deconvs, ...]
        (  # convs+deconvs: [nconvs+pooling, nconvs+concat]
            (  # nconvs+pooling: [nconvs, pooling]
                [  # nconvs: [conv, conv, ...]
                    (64, 3, "relu"),  # conv: [f, s, a]
                    (64, 3, "relu")
                ],
                "max"  # pooling
            ),
            (  # nconvs+concat: [nconvs, concat]
                [  # nconvs: [conv, conv, ...]
                    (64, 3, "relu"),  # conv: [f, s, a]
                    (64, 3, "relu")
                ],
                True  # concat
            )
        ),
        (
            (
                [
                    (128, 3, "relu"),
                    (128, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    (128, 3, "relu"),
                    (128, 3, "relu")
                ],
                True
            )
        ),
        (
            (
                [
                    (256, 3, "relu"),
                    (256, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    (256, 3, "relu"),
                    (256, 3, "relu")
                ],
                True
            )
        ),
        (
            (
                [
                    (512, 3, "relu"),
                    (512, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    (512, 3, "relu"),
                    (512, 3, "relu")
                ],
                True
            )
        )
    ],
    [  # bottleneck: [conv, conv, ...]
        (1024, 3, "relu"),  # conv: [f, s, a]
        (1024, 3, "relu")
    ]
)