/**
 * @file mujoco_pushing.cpp
 * @author your name (you@domain.com)
 * @brief 
 * create a planning scene in Mujoco. Push the object from start to goal.
 * use the planned trajectory to pass to Mujoco to validate.
 * @version 0.1
 * @date 2023-11-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

 #include "cmgmp/search/rrtplanner.h"
#include "cmgmp/utilities/sample_grasp.h"
#include "cmgmp/contacts/contact_kinematics.h"
#include "cmgmp/contacts/contact_utils.h"
#include "cmgmp/manipulators/DartPointManipulator.h"
#include "cmgmp/worlds/DartWorld.h"


#include <ctime>
#include <iostream>
#include <fstream>

#define VISUALIZE_SG 0
#define VISUALIZE_PTS 1
#define DO_SEARCH 2

void tabletop(Vector7d &x_start, Vector7d &x_goal, double &goal_thr,
               double &wa, double &wt, double &eps_trans, double &eps_angle,
               Vector3d &x_ub, Vector3d &x_lb, PlanningWorld *pw, std::vector<ContactPoint> *surface)
{
    DartWorld* my_world = new DartWorld();

    Vector3d box_size;
    box_size << 0.0175*2, 0.0175*2, 0.0475*2; // Dart: width, length, height. Mujoco: half size

    SkeletonPtr object = 
        createFreeBox("box_object", Vector3d(box_size));
    
    SkeletonPtr env_obj1 =
        createFixedBox("env_obj1", Vector3d(0.02*2,0.0775*2,0.075*2),
                       Vector3d(0.717423, 0.168684, 0.974784));
    SkeletonPtr workspace = 
        createFixedBox("workspace", Vector3d(0.25*2, 0.4*2, 0.5*2), 
                      Vector3d(0.2+0.67, 0, -0.1+0.5));


    // SkeletonPtr ground =
    //     createFixedBox("ground", Vector3d(10, 10, 1), Vector3d(0, 0, 1e-4 - 0.5));

    my_world->addObject(object);
    my_world->addEnvironmentComponent(env_obj1);
    my_world->addEnvironmentComponent(workspace);



    int n_robot_contacts = 1;
    DartPointManipulator *rpt =
        new DartPointManipulator(n_robot_contacts, 0.001);
    my_world->addRobot(rpt);
    rpt->is_patch_contact = true;

    pw->world = my_world;

    // parameters
    pw->mu_env = 0.1;
    pw->mu_mnp = 0.9;


    x_start << 0.953192, 0.270213, 0.947284-0.01, 0, 0, 0, 1;
    x_goal << 0.953192-0.05, 0.270213+0.04, 0.947284-0.01, 0, 0, 0, 1;


    x_ub << 0.25+0.2+0.67, 0.4, 0.947284+0.1;
    x_lb << -0.25+0.2+0.67, -0.4, 0.94;

    wa = 1.0;
    wt = 1.0;

    eps_trans = 0.001;
    eps_angle = 3.14 / 180;
    goal_thr = 0.01 * 3.14 * 10 / 180;

    // load object surface discretization;
    // MatrixXd data = load_points_from_csv(std::string(SRC_DIR) + "/data/grasp_sampling/bookshelf_surface_points.csv");
    // int N = data.rows();
    // for (int i = 0; i < N; ++i)  {
    //     ContactPoint pt((data.block(i,0,1,3)).transpose(), data.block(i,3,1,3).transpose(), 0);
    //     surface->push_back(pt);
    // }
    surfacepoints_from_file(std::string(SRC_DIR) + +"/data/grasp_sampling/box_halflength_1.csv", surface);
    for (std::vector<ContactPoint>::iterator it = surface->begin(); it != surface->end();)
    {
        if (it->n[1] < -0.8)
        {
            it = surface->erase(it);
        }
        else if (it->n[2] > 0.8){
            it = surface->erase(it);
        }
        else
        {
            ++it;
        }
    }
    for (int i = 0; i < surface->size(); i++)
    {
        surface->at(i).p = Vector3d(surface->at(i).p(0) * box_size[0] / 2, surface->at(i).p(1) * box_size[1] / 2, surface->at(i).p(2) * box_size[2] / 2);
    }
}


int main(int argc, char *argv[])
{

    RRTPlannerOptions options;
    options.goal_biased_prob = 0.7;
    options.max_samples = 500;
    // options.sampleSO3 = false;
    // options.rotation_sample_axis << 0, 0, 1;

    // int test_option = VISUALIZE_SG;
    int test_option = DO_SEARCH;

    Vector7d x_start;
    Vector7d x_goal;
    PlanningWorld pw;
    double wa;
    double wt;
    Vector3d x_ub;
    Vector3d x_lb;
    double eps_trans;
    double eps_angle;
    double goal_thr;
    std::vector<ContactPoint> surface;
    Vector6d f_w;
    f_w << 0, 0, -0.1, 0, 0, 0;

    tabletop(x_start, x_goal, goal_thr, wa, wt, eps_trans, eps_angle, x_ub, x_lb, &pw, &surface);

    if (test_option == VISUALIZE_SG)
    {
        std::vector<Vector7d> pp;
        pp.push_back(x_start);
        pp.push_back(x_goal);

        pw.world->setObjectTrajectory(pp);
    }
    else if (test_option == VISUALIZE_PTS)
    {
        Vector7d x;
        x << 0, 0, 0, 0, 0, 0, 1;
        pw.world->setSurfacePoints(surface); // TODO: draw contact points and their normals
    }
    else if (test_option == DO_SEARCH)
    {

        RRTPlanner planner(&pw);

        planner.Initialize(x_lb, x_ub, x_start, f_w, surface, wa, wt, eps_trans, eps_angle);

        // planner.SetInitialManpulatorLocations(mnps_config);

        std::vector<int> path;

        double t;
        bool success;

        planner.Search(options, x_goal, goal_thr, &path, success, t, false);
        double grasp_measure_charac_length = 1.0;
        planner.printResults(path, success, t, grasp_measure_charac_length);
        planner.VisualizePath(path);
        //
    }

    pw.world->startWindow(&argc, argv);
}