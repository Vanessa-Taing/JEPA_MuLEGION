import metaworld
import numpy as np

task_name="reach-v3"

mt1 = metaworld.MT1(task_name)

env = mt1.train_classes[task_name](
    render_mode="rgb_array"
)

env.set_task(mt1.train_tasks[0])

obs, info = env.reset()

frame = env.render()

print("Observation:", obs.shape)
print("Frame:", frame.shape)
print("dtype:", frame.dtype)

env.close()
