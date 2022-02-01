import tensorrt as trt


def create_example_plugin(scale):

    registry = trt.get_plugin_registry()

    creator = None
    for pc in registry.plugin_creator_list:
        if pc.name == 'ExamplePlugin':
            creator = pc
    
    fc = trt.PluginFieldCollection()
    fc.append(
        trt.PluginField(
            'scale',
            scale * np.ones((1,)).astype(np.float32),
            trt.PluginFieldType.FLOAT32
        )
    )

    return creator.create_plugin('ExamplePlugin', fc)