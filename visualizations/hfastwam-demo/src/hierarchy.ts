export interface ExampleFrame {
  frame: number
  title: string
  detail: string
  mainImage: string
  wristImage: string
  action: [number, number, number, number, number, number, number]
  proprio: [number, number, number, number, number, number, number, number]
}

export const actionDimensions = [
  'Δx',
  'Δy',
  'Δz',
  'Δroll',
  'Δpitch',
  'Δyaw',
  'grip',
] as const

export const realTrainingExample = {
  dataset: 'LIBERO Object · LeRobot',
  episode: 0,
  window: 'frames 112–144',
  task: 'pick up the alphabet soup and place it in the basket',
  plannedSubtask: 'place the grasped soup can into the basket',
  prompt:
    "A video recorded from a robot's point of view executing the following instruction: pick up the alphabet soup and place it in the basket",
  fps: 20,
  controlFrames: 33,
  actionHorizon: 32,
  sampledVideoFrames: 9,
  actionVideoRatio: 4,
  illustratedWorldActionCycles: 4,
  latentTargetsPerCycle: 2,
  actionTokensPerCycle: 32,
  frames: [
    {
      frame: 112,
      title: 'Lower',
      detail: 'Carry the grasped can down into the basket.',
      mainImage: 'libero-example/main_0.jpg',
      wristImage: 'libero-example/wrist_0.jpg',
      action: [-0.284, 0.0, -0.938, 0.06, -0.035, 0.001, 0.0],
      proprio: [-0.029, 0.217, 0.236, 3.103, -0.023, 0.313, 0.035, -0.028],
    },
    {
      frame: 124,
      title: 'Align',
      detail: 'Settle the object pose just above the receptacle.',
      mainImage: 'libero-example/main_1.jpg',
      wristImage: 'libero-example/wrist_1.jpg',
      action: [0.0, 0.174, -0.129, 0.0, 0.147, 0.0, 0.0],
      proprio: [-0.048, 0.225, 0.147, 3.15, -0.021, 0.308, 0.039, -0.024],
    },
    {
      frame: 132,
      title: 'Release',
      detail: 'Open the gripper and begin the upward retract.',
      mainImage: 'libero-example/main_2.jpg',
      wristImage: 'libero-example/wrist_2.jpg',
      action: [0.169, 0.005, 0.284, 0.0, 0.136, -0.045, 1.0],
      proprio: [-0.028, 0.234, 0.167, 3.172, -0.073, 0.112, 0.036, -0.027],
    },
    {
      frame: 140,
      title: 'Retract',
      detail: 'Move clear while the soup can remains in the basket.',
      mainImage: 'libero-example/main_3.jpg',
      wristImage: 'libero-example/wrist_3.jpg',
      action: [0.032, 0.038, 0.027, 0.0, 0.041, -0.006, 1.0],
      proprio: [-0.011, 0.236, 0.19, 3.169, -0.112, -0.078, 0.039, -0.038],
    },
  ] satisfies ExampleFrame[],
}
