/*
A 3D implementation of the SPH method using CUDA.

Based on the particles sample from the CUDA samples.

@author Octavio Navarro
@version 1.0
*/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <ctime>

#include <GL/freeglut.h>

#include "Solver.h"

using namespace std;

/// FPS calculation
char fps_message[50];
int frameCount = 0;
double average_fps = 0;
float fps = 0;
int total_fps_counts = 0;
int currentTime = 0, previousTime = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -100.0;

/// Keyboard controls
bool keypressed = false, simulate = true;

Solver *solver;

void calculateFPS()
{
	//  Increase frame count
	frameCount++;

	//  Get the number of milliseconds since glutInit called
	//  (or first call to glutGet(GLUT ELAPSED TIME)).
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	//  Calculate time passed
	int timeInterval = currentTime - previousTime;

	if (timeInterval > 1000)
	{
		//  calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);

		if(simulate)
		{
			average_fps += fps;
			total_fps_counts ++;
		}
		
		//  Set time
		previousTime = currentTime;

		//  Reset frame count
		frameCount = 0;
	}
}

void init(void)
{
	const GLubyte* renderer;
	const GLubyte* version;
	const GLubyte* glslVersion;

	renderer = glGetString(GL_RENDERER); /* get renderer string */
	version = glGetString(GL_VERSION); /* version as a string */
	glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
	printf("GLSL version supported %s\n", glslVersion);

	glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);

	solver = new Solver(NUM_PARTICLES);
}

void idle(void)
{
	if(simulate)
		solver->Update();

	calculateFPS();
	sprintf(fps_message, "SPH FPS %.3f", fps);
	glutSetWindowTitle(fps_message);

	glutPostRedisplay();
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

	/// Display Cube
	glPushMatrix();
		glColor3f(1,1,1);
		glScalef(GRID_SIZE, GRID_SIZE, GRID_SIZE);
		glutWireCube(1.0f);
	glPopMatrix();

	/// Display particles
	float3* positions = solver->GetPos();
	m3Vector* goalPositions = solver->GetGoalPos();

	for (int index = 0; index < NUM_PARTICLES; index++)
	{
		glPushMatrix();
			glBegin(GL_POINTS);
				glPointSize(10.0f);
				glColor3f(0.2, 0.2, 0.8);
				glVertex3f(positions[index].x-32, positions[index].y-32, positions[index].z-32);
			glEnd();
		glPopMatrix();

		// glPushMatrix();
		// 	glBegin(GL_POINTS);
		// 		glPointSize(10.0f);
		// 		glColor3f(0.2, 0.8, 0.2);
		// 		glVertex3f(goalPositions[index].x-32, goalPositions[index].y-32, goalPositions[index].z-32);
		// 	glEnd();
		// glPopMatrix();
	}

	glutSwapBuffers();
}

void reshape(int width, int height)
{
	glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, static_cast<float>(width) / (height = (height == 0 ? 1 : height)), 0.01, 1000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void exit_simulation()
{
	cout << "Average FPS: " << average_fps / total_fps_counts<< endl;
	exit(0);
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
		case 27:
			exit_simulation();
			break;
		case 32:
			simulate = !simulate;
			cout << "Simulation: " << ((simulate == true)? "On" : "Off") << endl;
			break;
	}
}


void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	glutInitWindowPosition(200,200);
	glutInitWindowSize(800, 600);
	glutCreateWindow("SPH");
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	init();

	glutMainLoop();

	return 0;
}