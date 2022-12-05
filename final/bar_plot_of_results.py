import matplotlib.pyplot as plt
import numpy as np

controller_types = ["LQR", "Discrete Value Iteration", "Discrete Actor Critic"]
seeds_to_index = {
    123: 0,
    42712: 1,
    27963: 2,
    6677: 3,
    8543: 4,
    7789: 5
}
systems_to_index = {
    "Linear System": 0,
    "Fluid Flow": 1,
    "Lorenz": 2,
    "Double Well": 3
}

# Columns are seeds
# Rows are systems

lqr_costs = np.array([
    [690.3651602503617, 655.9326140747207, 700.1022365832588, 732.7496556074263, 719.8111878208473, 721.8756381390517], # linear system
    [1609.8471808037928, 1600.055786203956, 1618.6768076115868, 1616.9956012694583, 1640.7010068234897, 1608.9899932500894], # fluid flow
    [1475234.6126, 1596213.9374344894, 1520834.7334292126, 1494760.3805688634, 1395727.3166997987, 1449749.8968759256], # lorenz
    [103704.51687117333, 97452.18065088477, 96464.2371536605, 100110.80969133413, 101984.64999562102, 100425.66115889452] # double well
])

discrete_value_iteration_costs = np.array([
    [746.1123197445295, 712.4402964082174, 756.8163778364084, 790.8962002388112, 778.2802228272905, 782.7128166924064], # linear system
    [1528.901923560299, 1542.7627740690978, 1535.4295134254392, 1543.6169267883058, 1541.4170214683143, 1527.266155876525], # fluid flow
    [160554.62643234286, 327396.58141615504, 205494.2987618653, 86419.59531588468, 271680.2392170337, 359468.6340085569], # lorenz
    [2661.3981326371186, 2649.8719426954235, 2655.777440909467, 2663.5144408863794, 2645.504256670517, 2662.120564554043] # double well
])

discrete_actor_critic_costs = np.array([
    [3558.0384877681104, 832.287110117803, 4040.3280498945037, 6589202.4515740285, 18677.156457102217, 1127.3739302275922], # linear system
    [900.5468933156702, 881.905980560978, 818.0213603314646, 899.5495994555026, 907.4898088374696, 909.9871054735924], # fluid flow
    [163894.15747720585, 160258.34251511592, 156640.41452218516, 1665740.353984627, 1577904.108433765, 1593290.5333130744], # lorenz
    [2334.8643608366597, 2328.8439248476193, 2418.543277588186, 2409.4575622218654, 2394.0184537442706, 2483.8141637182093] # double well
])

results = np.array([lqr_costs, discrete_value_iteration_costs, discrete_actor_critic_costs])

# Plot for each system
for system_name in systems_to_index.keys():
    fig, ax = plt.subplots()
    ax.set_title(f"Average Cost Over 100 Episodes - {system_name}")
    ax.set_xlabel("Controller Types")
    ax.set_ylabel("Average Cost")
    bars = ax.bar(
        controller_types,
        results[:,systems_to_index[system_name]].mean(axis=1)
    )
    ax.bar_label(bars)
    file_name = f"Average-Cost-{'-'.join(system_name.split(' '))}"
    plt.savefig(file_name + '.png')
    plt.savefig(file_name + '.svg')
    plt.show()
