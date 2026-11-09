import glob
import os
import runpy
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size' : 11})


Re_100_dict = {'CFF' : [[], [], []], 'Bouzidi' : [[], [], []], 'Chen' : [[], [], []]} # [R, C_D, x_s]
CFF_dict = {}
for Re_folder in glob.glob('./Re*/'):
    
    Re = int(Re_folder.split("/")[-2].replace("Re", ""))

    
    for R_folder in glob.glob(f'{Re_folder}/R*/'):
        R = int(R_folder.split("/")[-2].replace("R", ""))

        if R not in CFF_dict:
            CFF_dict[R] = {"Re" : [], "C_D" : [], "x_s" : [], "separation_angle" : []}

        Ftot_file = f"{R_folder}/output/CFF/Ftot.csv"
        if not os.path.isfile(Ftot_file):
            continue

        cells_per_radius = runpy.run_path(f"{R_folder}/resolution_definition.py")['cells_per_radius']
        Re = runpy.run_path(f"{Re_folder}/Reynolds_definition.py")['Re']
        case = runpy.run_path("./case_definition.py")['Case'](Re=Re, cells_per_radius=cells_per_radius)
        ux_inf = case.ux_inf

        # Load the drag through time
        Ftot_data = np.genfromtxt(Ftot_file, skip_header=1, delimiter=",")
        if Ftot_data.size == 0:
            continue
        D_arr = Ftot_data[:,1]
        # assert abs(np.diff(D_arr[-2:])/D_arr[-1]) < 0.01, 'Drag force not yet converged'

        # Calculate drag coefficient at final time step
        D = D_arr[-1]
        C_D = 2*D / (R**2 * np.pi * ux_inf**2)
        
        # Load separation length
        x_s_file = f"{R_folder}/output/CFF/separation_length.csv"
        if not os.path.isfile(x_s_file):
            continue
        x_s_data = np.genfromtxt(x_s_file, skip_header=1, delimiter=",")
        x_s = x_s_data.item()

        # Load separation angle
        separation_angle_file = f"{R_folder}/output/CFF/separation_angle.csv"
        if not os.path.isfile(separation_angle_file):
            continue
        separation_angle_data = np.genfromtxt(separation_angle_file, skip_header=1, delimiter=",")
        separation_angle = separation_angle_data.item()

        CFF_dict[R]["Re"].append(Re)
        CFF_dict[R]["C_D"].append(C_D)
        CFF_dict[R]["x_s"].append(x_s)
        CFF_dict[R]["separation_angle"].append(separation_angle)

        if Re == 100:
            Re_100_dict['CFF'][0].append(R)
            Re_100_dict['CFF'][1].append(C_D)
            Re_100_dict['CFF'][2].append(x_s)

Bouzidi_dict = {}
for R_folder in glob.glob(f'./Re100/R*/'):
    R = int(R_folder.split("/")[-2].replace("R", ""))
    
    if R not in Bouzidi_dict:
        Bouzidi_dict[R] = {"Re" : [100], "C_D" : [], "x_s" : [], "separation_angle" : []}
    

    Ftot_file = f"{R_folder}/output/Bouzidi/Ftot.csv"
    if not os.path.isfile(Ftot_file):
        continue

    cells_per_radius = runpy.run_path(f"{R_folder}/resolution_definition.py")['cells_per_radius']
    Re = runpy.run_path(f"{Re_folder}/Reynolds_definition.py")['Re']
    case = runpy.run_path("./case_definition.py")['Case'](Re=Re, cells_per_radius=cells_per_radius)
    ux_inf = case.ux_inf

    # Load the drag through time
    Ftot_data = np.genfromtxt(Ftot_file, skip_header=1, delimiter=",")
    if Ftot_data.size == 0:
        continue
    D_arr = Ftot_data[:,1]
    # assert abs(np.diff(D_arr[-2:])/D_arr[-1]) < 0.01, 'Drag force not yet converged'

    # Calculate drag coefficient at final time step
    D = D_arr[-1]
    C_D = 2*D / (R**2 * np.pi * ux_inf**2)
    
    # Load separation length
    x_s_file = f"{R_folder}/output/Bouzidi/separation_length.csv"
    if not os.path.isfile(x_s_file):
        continue
    x_s_data = np.genfromtxt(x_s_file, skip_header=1, delimiter=",")
    x_s = x_s_data.item()

    
    Bouzidi_dict[R]["C_D"].append(C_D)
    Bouzidi_dict[R]["x_s"].append(x_s)

    Re_100_dict['Bouzidi'][0].append(R)
    Re_100_dict['Bouzidi'][1].append(C_D)
    Re_100_dict['Bouzidi'][2].append(x_s)


Chen_dict = {}
for R_folder in glob.glob(f'./Re100/R*/'):
    R = int(R_folder.split("/")[-2].replace("R", ""))
    
    if R not in Chen_dict:
        Chen_dict[R] = {"Re" : [100], "C_D" : [], "x_s" : [], "separation_angle" : []}
    

    Ftot_file = f"{R_folder}/output/Chen/Ftot.csv"
    if not os.path.isfile(Ftot_file):
        continue

    cells_per_radius = runpy.run_path(f"{R_folder}/resolution_definition.py")['cells_per_radius']
    Re = runpy.run_path(f"{Re_folder}/Reynolds_definition.py")['Re']
    case = runpy.run_path("./case_definition.py")['Case'](Re=Re, cells_per_radius=cells_per_radius)
    ux_inf = case.ux_inf

    # Load the drag through time
    Ftot_data = np.genfromtxt(Ftot_file, skip_header=1, delimiter=",")
    if Ftot_data.size == 0:
        continue
    D_arr = Ftot_data[:,1]
    # assert abs(np.diff(D_arr[-2:])/D_arr[-1]) < 0.01, 'Drag force not yet converged'

    # Calculate drag coefficient at final time step
    D = D_arr[-1]
    C_D = 2*D / (R**2 * np.pi * ux_inf**2)

    # Load separation length
    x_s_file = f"{R_folder}/output/Chen/separation_length.csv"
    if not os.path.isfile(x_s_file):
        continue
    x_s_data = np.genfromtxt(x_s_file, skip_header=1, delimiter=",")
    x_s = x_s_data.item()
    
    Chen_dict[R]["C_D"].append(C_D)
    Chen_dict[R]["x_s"].append(x_s)

    Re_100_dict['Chen'][0].append(R)
    Re_100_dict['Chen'][1].append(C_D)
    Re_100_dict['Chen'][2].append(x_s)


# Load reference data
ref_dict = {}
ref_dict['separation_angle'] = np.genfromtxt("../Ref/separation_angle_vs_Reynolds.csv", skip_header=1, delimiter=",").T
ref_dict['C_D'] = np.genfromtxt("../Ref/C_D_vs_Reynolds.csv", skip_header=0, delimiter=",").T
ref_dict['x_s'] = np.genfromtxt("../Ref/separation_length_vs_Reynolds.csv", skip_header=1, delimiter=",").T


# Plotting
markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange']
ylabels = {"C_D" : r"$C_D$ [-]", "x_s" : r"$x_s/D$ [-]", "separation_angle" : r"$\Theta_s$ (degrees)"}
for var in ["C_D", "x_s", "separation_angle"]:
    plt.figure(figsize=(5, 7/2))
    for i, R in enumerate(sorted(CFF_dict.keys())):

        sort_inds = np.argsort(CFF_dict[R]["Re"])

        plt.plot(np.array(CFF_dict[R]["Re"])[sort_inds],
                 np.array(CFF_dict[R][var])[sort_inds],
                 label=f'{2*R} cpd', zorder=20,
                 marker=markers[i % len(markers)], markersize=5,
                 color=colors[i % len(colors)])

    # Plot reference data, italic label
    plt.plot(ref_dict[var][0], ref_dict[var][1], label='Ref.', zorder=10, color='k', linestyle='--')
    
    plt.xlabel('Reynolds Number [-]')
    plt.ylabel(ylabels[var])
    legend = plt.legend(ncol=2)
    # Opaque legend
    legend.get_frame().set_alpha(1)
    legend.set_zorder(30)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./Figures/{var}.pdf', dpi=300)


# Plot absolute value against resolution for the three methods for Re=100
for var_ind, var in zip([1, 2], ["C_D", "x_s"]):
    plt.figure(figsize=(5, 7/2))
    sort_inds_CFF = np.argsort(Re_100_dict['CFF'][0])
    plt.plot(2*np.array(Re_100_dict['CFF'][0])[sort_inds_CFF], np.array(Re_100_dict['CFF'][var_ind])[sort_inds_CFF], label='CFF', zorder=20,
            color='r')
    sort_inds_Bouzidi = np.argsort(Re_100_dict['Bouzidi'][0])
    plt.plot(2*np.array(Re_100_dict['Bouzidi'][0])[sort_inds_Bouzidi], np.array(Re_100_dict['Bouzidi'][var_ind])[sort_inds_Bouzidi], label='Bouzidi', zorder=20,
            linestyle='dashdot', color='g')
    sort_inds_Chen = np.argsort(Re_100_dict['Chen'][0])
    plt.plot(2*np.array(Re_100_dict['Chen'][0])[sort_inds_Chen], np.array(Re_100_dict['Chen'][var_ind])[sort_inds_Chen], label='Chen', zorder=20,
            linestyle=(0,(1,1)), color='b')
    plt.axhline(ref_dict[var][1][2], label='Ref.', zorder=10, color='k', linestyle='--')
    plt.xlabel('Cells per diameter [-]')
    plt.ylabel(f'{ylabels[var]}')
    plt.xscale('log')
    # Set xticks and labels to show only the R values used
    # Also remove minor xticks
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    plt.xticks([4, 8, 16, 32, 64], [str(x) for x in [4, 8, 16, 32, 64]])
    legend = plt.legend(ncol=2)
    # Opaque legend
    legend.get_frame().set_alpha(1)
    legend.set_zorder(30)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./Figures/Re100_{var}_abs_convergence.pdf')



# Plot difference between subsequent resolutions
# Plot error against resolution for various Reynolds numbers
diff_dict = {}
for var_ind, var in zip([1, 2], ["C_D", "x_s"]):
    diff_dict[var] = {}

    sorted_Rs = sorted(CFF_dict.keys())
    for i, R in enumerate(sorted_Rs[1:], start=1):
        R_prev = sorted_Rs[i-1]

        for Re in np.intersect1d(CFF_dict[R]["Re"], CFF_dict[R_prev]["Re"]):
            if Re not in diff_dict[var]:
                diff_dict[var][Re] = [[], []] # [R, var_diff]
            
            diff_dict[var][Re][0].append(R)
            diff_dict[var][Re][1].append(abs(CFF_dict[R][var][CFF_dict[R]["Re"].index(Re)] - CFF_dict[R_prev][var][CFF_dict[R_prev]["Re"].index(Re)]))
        


    plt.figure(figsize=(5, 5))
    for i, (Re, (R_list, diff_list)) in enumerate(diff_dict[var].items()):        
        plt.loglog(2*np.array(R_list), diff_list, label=f'$Re = {Re}$', zorder=20,
                 marker=markers[i % len(markers)], markersize=5,
                 color=colors[i % len(colors)])

    # Plot Bouzidi
    sort_inds_Bouzidi = np.argsort(Re_100_dict['Bouzidi'][0])
    bouzidi_sorted = np.array(Re_100_dict['Bouzidi'][var_ind])[sort_inds_Bouzidi]
    diff_bouzidi = abs(bouzidi_sorted[1:] - bouzidi_sorted[:-1])
    plt.loglog(2*np.array(Re_100_dict['Bouzidi'][0])[sort_inds_Bouzidi][1:], diff_bouzidi, label='Bouzidi Re=100', zorder=20, lw=2,
            color='g', linestyle='dashdot')

    # Plot Chen
    sort_inds_Chen = np.argsort(Re_100_dict['Chen'][0])
    chen_sorted = np.array(Re_100_dict['Chen'][var_ind])[sort_inds_Chen]
    diff_chen = abs(chen_sorted[1:] - chen_sorted[:-1])
    plt.loglog(2*np.array(Re_100_dict['Chen'][0])[sort_inds_Chen][1:], diff_chen, label='Chen Re=100', zorder=20, lw=2,
            color='b', linestyle=(0,(1,1)))

    # Plot x and x^2 slopes for reference
    plt.axline((diff_dict[var][100][0][0], diff_dict[var][100][1][0]), 
               (diff_dict[var][100][0][0]*2, diff_dict[var][100][1][0]/2), 
               color='k', linestyle=(0, (1, 3)), label=r'$x$')
    plt.axline((2*diff_dict[var][100][0][0], diff_dict[var][100][1][0]), 
               (2*diff_dict[var][100][0][0]*2, diff_dict[var][100][1][0]/4), 
               color='k', linestyle='dashed', label=r'$x^2$')

    plt.xlim(7, 70)
    plt.ylim(1e-4, 1)
    plt.xlabel('Cells per diameter [-]')
    plt.ylabel(f'Difference in {ylabels[var]}')
    # Set xticks and labels to show only the R values used
    # Also remove minor xticks
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    plt.xticks(2*np.array(sorted_Rs[1:]), [str(2*R) for R in sorted_Rs[1:]])
    legend = plt.legend(ncol=2)
    # Opaque legend
    legend.get_frame().set_alpha(1)
    legend.set_zorder(30)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./Figures/diff_{var}.pdf', dpi=300)


plt.show()
