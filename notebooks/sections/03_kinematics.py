# %% [markdown]
# ## 3 · Follow one leg through the kinematic tree
#
# The floating base describes the body's position and orientation; it is not a
# motor. Each leg branches through **hip yaw → hip roll → hip pitch → knee → ankle**.
# The neck/head is another branch. The tree below is read from the actual model;
# fixed connections are collapsed to keep it readable.

# %% [markdown]
# ### Instructor example · joints on the actual model
#
# | Joint centers through the covers | The same chain in the model |
# |:---:|:---:|
# | ![Left-leg joints on BDX-R](lesson-asset:reference/annotated_robot.png) | ![Articulated kinematic tree](lesson-asset:reference/kinematic_tree.png) |
#
# Follow matching colors and numbers. Dots mark the model's internal joint centers;
# translucent covers make hidden joints visible. Other tree branches are collapsed.
#
# ### Your activity · inspect joints, then move one
#
# Run the inspection to print joint axes and limits and regenerate these figures.
# Then use the knee slider below to compare your prediction with forward kinematics.

# %%
%%robot
from bdxr_classroom.lesson.kinematics import (
    print_joint_table,
    save_tree,
    save_annotated_pose,
)

print_joint_table(env)
save_tree(env, "figures/kinematic_tree.svg")
save_annotated_pose(env, "figures/annotated_robot.png")

# %%
media_pair(
    [
        (
            "Follow the actual left-leg joints",
            WORK / "figures/annotated_robot.png",
            "Inspect the hip, knee, and ankle on one leg.",
        ),
        (
            "Actual articulated joint tree",
            WORK / "figures/kinematic_tree.svg",
            "Left and right legs have five actuated joints each. The head/neck has four.",
        ),
    ]
)

# %% [markdown]
# ### Joint angles determine foot position
#
# Let $\mathbf q_L$ contain the five left-leg joint angles. Forward kinematics
# computes the foot position relative to the torso:
#
# $$\mathbf p_{\mathrm{foot}} = f(\mathbf q_L), \qquad
# {}^{B}T_F = {}^{B}T_1(q_1)\,{}^{1}T_2(q_2)\cdots{}^{5}T_F.$$
#
# Each transform contributes a joint rotation and a fixed link offset. The local
# rotation axes and limits are listed above. Mirrored legs can use opposite axis
# signs—do not assume that equal left/right angles produce identical motion.
#
# **Predict:** holding all other angles fixed, what path will the foot trace as
# the knee turns? Is the path a straight line?

# %% [markdown]
# | One-joint forward kinematics |
# |:---:|
# | ![Measured knee-sweep foot path](lesson-asset:reference/knee_path.png) |
#
# **Instructor example:** measured positions from −20° to +40° of left-knee motion,
# with the root and all other joints fixed. The curve is geometry, not a stable gait.
# Run the next cells to generate your own interactive view and change the angle range.

# %%
%%robot
from bdxr_classroom.lesson.kinematics import sweep_joint

# Angles are in degrees here for readability; the model uses radians internally.
angles = range(-20, 41, 5)
foot_positions = sweep_joint(env, "Left_Knee", angles, "figures/knee_sweep.html")
print("Generated", len(foot_positions), "forward-kinematics configurations.")

# %%
from bdxr_classroom.lesson.presentation import interactive_html

interactive_html(WORK / "figures/knee_sweep.html", height=610)

# %% [markdown]
# The slider changes one joint with the root and other joints fixed. The foot
# coordinates are measured in the torso frame. This is **kinematics**, not a
# demonstration that every pose is balanced. It traces one joint's motion,
# not the robot's full reachable workspace.
#
# Try a smaller angle interval or another left-leg joint within its listed limits.
# Compare the geometric effect before changing any training reward.
