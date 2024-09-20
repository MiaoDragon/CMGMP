/**
 * @file mujoco_pushing_test.cpp
 * @author your name (you@domain.com)
 * @brief 
 * use mujoco to load scene, and then get the object etc
 * to transfer to CMGMP format. After planning, get the
 * output to visualize in mujoco, and execute.
 *
 * @version 0.1
 * @date 2023-11-09
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

#include <mujoco/mjmodel.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjvisualize.h>
#include <GLFW/glfw3.h>



#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>


#define VISUALIZE_SG 0
#define VISUALIZE_PTS 1
#define DO_SEARCH 2


// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;






void tabletop(int obj_body_id, int robot_body_id, const std::vector<int>& env_body_ids,
              int obj_body_qadr, int obj_body_dofadr, 
              int robot_body_jntadr, int robot_body_dofadr,
              Vector7d &x_start, Vector7d &x_goal, double &goal_thr,
              double &wa, double &wt, double &eps_trans, double &eps_angle,
              Vector3d &x_ub, Vector3d &x_lb, PlanningWorld *pw, std::vector<ContactPoint> *surface)
{
    DartWorld* my_world = new DartWorld();

    Vector3d obj_box_size;
    // get the object body shape
    int obj_geomadr = m->body_geomadr[obj_body_id];
    double obj_size_x = m->geom_size[obj_geomadr*3+0]*2;
    double obj_size_y = m->geom_size[obj_geomadr*3+1]*2;
    double obj_size_z = m->geom_size[obj_geomadr*3+2]*2;

    double obj_pos_x = m->body_pos[obj_body_id*3+0] + m->geom_pos[obj_geomadr*3+0];
    double obj_pos_y = m->body_pos[obj_body_id*3+1] + m->geom_pos[obj_geomadr*3+1];
    double obj_pos_z = m->body_pos[obj_body_id*3+2] + m->geom_pos[obj_geomadr*3+2];

    double obj_quat_w = m->body_quat[obj_body_id*4+0];
    double obj_quat_x = m->body_quat[obj_body_id*4+1];
    double obj_quat_y = m->body_quat[obj_body_id*4+2];
    double obj_quat_z = m->body_quat[obj_body_id*4+3];



    obj_box_size << obj_size_x, obj_size_y, obj_size_z; // Dart: width, length, height. Mujoco: half size

    SkeletonPtr object = 
        createFreeBox("box_object", Vector3d(obj_box_size));

    my_world->addObject(object);

    for (int i=0; i<env_body_ids.size(); i++)
    {
        int geomadr = m->body_geomadr[env_body_ids[i]];
        double env_size_x = m->geom_size[geomadr*3+0]*2;
        double env_size_y = m->geom_size[geomadr*3+1]*2;
        double env_size_z = m->geom_size[geomadr*3+2]*2;
        double env_pos_x = m->body_pos[env_body_ids[i]*3+0] + m->geom_pos[geomadr*3+0];
        double env_pos_y = m->body_pos[env_body_ids[i]*3+1] + m->geom_pos[geomadr*3+1];
        double env_pos_z = m->body_pos[env_body_ids[i]*3+2] + m->geom_pos[geomadr*3+2];

        SkeletonPtr envi = 
            createFixedBox("envi", Vector3d(env_size_x, env_size_y, env_size_z), 
                        Vector3d(env_pos_x, env_pos_y, env_pos_z));
        my_world->addEnvironmentComponent(envi);

    }
    
    int n_robot_contacts = 1;
    DartPointManipulator *rpt =
        new DartPointManipulator(n_robot_contacts, 0.001);
    my_world->addRobot(rpt);
    rpt->is_patch_contact = false;  // three-point approx. of contact

    pw->world = my_world;

    // parameters
    pw->mu_env = 0.8;
    pw->mu_mnp = 0.01;


    x_start << obj_pos_x, obj_pos_y, obj_pos_z, obj_quat_x, obj_quat_y, obj_quat_z, obj_quat_w;
    x_goal << obj_pos_x - 0.04, obj_pos_y + 0.05, obj_pos_z, obj_quat_x, obj_quat_y, obj_quat_z, obj_quat_w;


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
        if (it->n[2] < -0.8)
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
        surface->at(i).p = Vector3d(surface->at(i).p(0) * obj_box_size[0] / 2, 
                                    surface->at(i).p(1) * obj_box_size[1] / 2, 
                                    surface->at(i).p(2) * obj_box_size[2] / 2);
    }

    // disable [0,0,+-1] since we need to reason about force if applying point contact at those locations.
    // given only pose information, we can't move the object.
}




// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}



int main(int argc, char *argv[])
{
    // read mujoco scene
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/motoman_ws/src/pracsys_vbnpm/tests/push_trial_3.xml";
    const char* c = filename.c_str();
    char loadError[1024] = "";

    m = mj_loadXML(c, 0, loadError, 1024);
    if (!m)
    {
        mju_error("Could not init model");
    }
    d = mj_makeData(m);

    if (!glfwInit())
    {
        mju_error("Could not init glfw");
    }

    // define object vars
    std::string obj_name = "object_body_0";
    std::string robot_name = "robot";
    int obj_body_id;
    int robot_body_id;
    std::vector<int> env_body_ids;  // each one has one block geom

    int obj_body_qadr;
    int obj_body_dofadr;  // obj has dof num 6, q num 7

    int robot_body_jntadr;
    // int robot_body_jntnum;
    int robot_body_dofadr;
    // int robot_body_dofnum;

    /* load the scene spec */
    for (int i=0; i<m->nbody; i++)
    {
        // name of the body
        const char* name = mj_id2name(m, mjOBJ_BODY, i);
        std::string name_s(name);
        // check if the body is object, env or robot
        if (name_s == obj_name)
        {
            obj_body_id = i;
            obj_body_qadr = m->jnt_qposadr[m->body_jntadr[i]];
            obj_body_dofadr = m->body_dofadr[i];
        }
        else if (name_s == robot_name)
        {
            robot_body_id = i;
            robot_body_jntadr = m->body_jntadr[i];
            std::cout << "robot body jnt_num: " << m->body_jntnum[i] << std::endl;

            const char* name1 = mj_id2name(m, mjOBJ_JOINT, m->body_jntadr[i]);
            std::string name1_s(name1);
            const char* name2 = mj_id2name(m, mjOBJ_JOINT, m->body_jntadr[i]+1);
            std::string name2_s(name2);
            const char* name3 = mj_id2name(m, mjOBJ_JOINT, m->body_jntadr[i]+2);
            std::string name3_s(name3);

            std::cout << "joint 0: " << name1_s << std::endl;
            std::cout << "joint 1: " << name2_s << std::endl;
            std::cout << "joint 2: " << name3_s << std::endl;

            // check if the name starts with "object"
            std::string name_s(name);
            // robot_body_jntnum = m->body_jntnum[i];
            robot_body_dofadr = m->body_dofadr[i];
            // robot_body_dofnum = m->body_dofnum[i];

            // print out the info
        }
        else
        {
            env_body_ids.push_back(i);
        }
    }

    /* load the scene in CMGMP */
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


    // load start and goal (pos, ori), CMGMP ori = qx,qy,qz,qw
    // mujoco: w,qx,qy,qz



    // pw.cons = new ContactConstraints(4);


    tabletop(obj_body_id, robot_body_id, env_body_ids, obj_body_qadr, obj_body_dofadr,
             robot_body_jntadr, robot_body_dofadr,
             x_start, x_goal, goal_thr, wa, wt, eps_trans, eps_angle, x_ub, x_lb,
             &pw, &surface);

    RRTPlannerOptions options;
    options.goal_biased_prob = 0.7;
    options.max_samples = 500;

    RRTPlanner planner(&pw);
    std::vector<int> node_path;

    // options.sampleSO3 = false;
    // options.rotation_sample_axis << 0, 0, 1;

    // int test_option = VISUALIZE_SG;
    int test_option = DO_SEARCH;


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


        planner.Initialize(x_lb, x_ub, x_start, f_w, surface, wa, wt, eps_trans, eps_angle);

        // planner.SetInitialManpulatorLocations(mnps_config);

        double t;
        bool success;

        planner.Search(options, x_goal, goal_thr, &node_path, success, t, false);
        double grasp_measure_charac_length = 1.0;
        planner.printResults(node_path, success, t, grasp_measure_charac_length);
        planner.VisualizePath(node_path);
        //
    }


    std::vector<Vector7d> object_poses;
    std::vector<VectorXd> mnp_configs;

    for (auto &k : node_path) {
        if (k == 0) {
        continue;
        }
        std::vector<Vector7d> path = planner.T->edges[planner.T->nodes[k].edge].path;
        VectorXd mnp_config = planner.T->nodes[k].manipulator_config;
        // std::cout << mnp_config.transpose() << std::endl;
        for (auto x : path) {
        object_poses.push_back(x);
        mnp_configs.push_back(mnp_config);
        }
    }

    std::cout << "object poses: " << std::endl;
    for (int i=0; i<object_poses.size(); i++)
    {
        std::cout << object_poses[i].transpose() << std::endl;
    }

    std::cout << "mnp_configs: " << std::endl;
    for (int i=0; i<mnp_configs.size(); i++)
    {
        std::cout << mnp_configs[i].transpose() << std::endl;
    }


    // mnp_config: p, n in the object frame
    // pw.world->startWindow(&argc, argv);

    std::vector<Vector3d> mnp_poses;
    // obtain the manipulator position in the world
    for (int i=0; i<mnp_configs.size(); i++)
    {
        Vector7d pose = object_poses[i];  // qx,qy,qz,qw
        // qx,qy,qz,qw -> qw,qx,qy,qz
        pose[3] = object_poses[i][6];
        pose[4] = object_poses[i][3];
        pose[5] = object_poses[i][4];
        pose[6] = object_poses[i][5];
        Matrix4d T = pose2SE3(pose.data());  // qw,qx,qy,qz
        // std::cout << "pose[" << i << "]: " << object_poses[i].transpose() << std::endl;
        // std::cout << "T[" << i << "]: " << T << std::endl;
        mnp_poses.push_back(T.block(0,0,3,3)*mnp_configs[i].head(3) + T.block(0,3,3,1));
    }


    // init GLFW, create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);


    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    int traj_idx = 0;
    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        if (traj_idx < object_poses.size())
        {
            // set the poses of the object
            // stored quat: qx,qy,qz,qw
            // mujoco: qw,qx,qy,qz
            d->qpos[obj_body_qadr+0] = object_poses[traj_idx][0];
            d->qpos[obj_body_qadr+1] = object_poses[traj_idx][1];
            d->qpos[obj_body_qadr+2] = object_poses[traj_idx][2];
            d->qpos[obj_body_qadr+3] = object_poses[traj_idx][6];
            d->qpos[obj_body_qadr+4] = object_poses[traj_idx][3];
            d->qpos[obj_body_qadr+5] = object_poses[traj_idx][4];
            d->qpos[obj_body_qadr+6] = object_poses[traj_idx][5];

            // set the positions of the manipulator
            d->qpos[m->jnt_qposadr[robot_body_jntadr+0]] = mnp_poses[traj_idx][0];
            d->qpos[m->jnt_qposadr[robot_body_jntadr+1]] = mnp_poses[traj_idx][1];
            d->qpos[m->jnt_qposadr[robot_body_jntadr+2]] = mnp_poses[traj_idx][2];


            // mj_forward(m, d);
            mj_step(m, d);

            traj_idx += 1;
        }
        else
        {
            traj_idx = traj_idx % object_poses.size();
        }
        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0/60.0)
        {
            // mj_step(m, d);
        }

        // get framebuffer viewport
        mjrRect viewport = {0,0,0,0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffer
        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    // free vis storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free mujoco model and data
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;


}
