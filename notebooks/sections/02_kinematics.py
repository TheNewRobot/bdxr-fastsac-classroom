# %% [markdown]
# ## 2 · Explore the robot’s kinematics
#
# The floating base describes the body's position and orientation; it is not a
# motor. Each leg branches through **hip yaw → hip roll → hip pitch → knee → ankle**.
# The neck/head is another branch. The tree below is read from the actual model;
# fixed connections are collapsed to keep it readable.

# %% [markdown]
# ### Instructor example · joints on the actual model
#
# | Left-leg links and joint pivots | The same chain in the model |
# |:---:|:---:|
# | ![Left-leg joints on BDX-R](lesson-asset:reference/annotated_robot.png) | ![Articulated kinematic tree](lesson-asset:reference/kinematic_tree.png) |
#
# The left leg faces the camera and is highlighted. Colored collars mark its
# measured hinge centers; these are schematic overlays, not motor CAD.
# Match their numbers to the tree. Other branches are faded or collapsed.
#
# ### Your activity · inspect joints, then move one
#
# Run the inspection to read the actual joint axes and limits.
# Then use the knee slider below to compare your prediction with forward kinematics.

# %%
%%robot
from bdxr_classroom.lesson.kinematics import print_joint_table

print_joint_table(env)

# %% [markdown]
# ### Joint angles determine foot position
#
# For the slider, we view the **right leg** from its visible side. Let
# $\mathbf q_R$ contain its five joint angles. Forward kinematics
# computes the foot position relative to the torso:
#
# $$\mathbf p_{\mathrm{foot}} = f(\mathbf q_R), \qquad
# {}^{B}T_F = {}^{B}T_1(q_1)\,{}^{1}T_2(q_2)\cdots{}^{5}T_F.$$
#
# Joint limits restrict which angles a motor can reach. The joint axis tells us
# which direction it rotates about; the link geometry sets the distance to the
# next joint. A hinge angle alone is therefore insufficient to locate the foot.
# The torso frame provides a common origin for comparing foot positions.
#
# Each transform contributes a joint rotation and a fixed link offset. The local
# rotation axes and limits are listed above. Mirrored legs can use opposite axis
# signs—do not assume that equal left/right angles produce identical motion.
#
# **Predict:** holding all other angles fixed, what path will the foot trace as
# the knee turns? Is the path a straight line?

# %% [markdown]
# The slider below moves the right knee with the root and other joints fixed.
# Predict its foot path, then generate the interactive view to check your answer.

# %%
%%robot
from bdxr_classroom.lesson.kinematics import sweep_joint

# Angles are in degrees here for readability; the model uses radians internally.
angles = range(-40, 21, 5)
foot_positions = sweep_joint(env, "Right_Knee", angles, "figures/knee_sweep.html")
print("Generated", len(foot_positions), "forward-kinematics configurations.")

# %%
from bdxr_classroom.lesson.presentation import joint_explorer

explorer = joint_explorer(WORK / "figures/knee_sweep.html")

# %% [markdown]
# The slider changes one joint with the root and other joints fixed. The foot
# coordinates are measured in the torso frame. This is **kinematics**, not a
# demonstration that every pose is balanced. It traces one joint's motion,
# not the robot's full reachable workspace.
#
# Try a smaller angle interval or another right-leg joint within its listed limits.
# Compare the geometric effect before changing any training reward.
