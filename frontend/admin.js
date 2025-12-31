const API_URL = '/api/admin';

// DOM Elements
let courseList, courseUrl, sessionCookie, courseContent;
let scrapeBtn, uploadBtn, statusModal, statusMessage, loadingSpinner, closeModalBtn;

document.addEventListener('DOMContentLoaded', () => {
    initElements();
    setupEventListeners();
    loadCourses();
});

function initElements() {
    courseList = document.getElementById('courseList');
    courseUrl = document.getElementById('courseUrl');
    sessionCookie = document.getElementById('sessionCookie');
    courseContent = document.getElementById('courseContent');
    scrapeBtn = document.getElementById('scrapeBtn');
    uploadBtn = document.getElementById('uploadBtn');
    statusModal = document.getElementById('statusModal');
    statusMessage = document.getElementById('statusMessage');
    loadingSpinner = document.getElementById('loadingSpinner');
    closeModalBtn = document.getElementById('closeModal');
}

function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    scrapeBtn.addEventListener('click', scrapeCourse);
    uploadBtn.addEventListener('click', uploadCourse);
    closeModalBtn.addEventListener('click', hideModal);

    // Close modal on outside click
    statusModal.addEventListener('click', (e) => {
        if (e.target === statusModal) {
            hideModal();
        }
    });

    // Close modal on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !statusModal.classList.contains('hidden')) {
            hideModal();
        }
    });
}

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}Tab`).classList.add('active');
}

async function loadCourses() {
    try {
        courseList.innerHTML = '<p class="no-courses">Loading courses...</p>';

        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) {
            throw new Error('Failed to fetch courses');
        }

        const data = await response.json();
        renderCourseList(data.courses);
    } catch (error) {
        courseList.innerHTML = '<p class="no-courses" style="color: #f87171;">Failed to load courses</p>';
        console.error('Error loading courses:', error);
    }
}

function renderCourseList(courses) {
    if (!courses || courses.length === 0) {
        courseList.innerHTML = '<p class="no-courses">No courses indexed yet</p>';
        return;
    }

    courseList.innerHTML = courses.map(course => `
        <div class="course-card">
            <div class="course-header">
                <h3>${escapeHtml(course.title)}</h3>
                <button class="delete-btn" onclick="deleteCourse('${escapeHtml(course.title).replace(/'/g, "\\'")}')">
                    Delete
                </button>
            </div>
            <div class="course-meta">
                <span>Instructor: ${escapeHtml(course.instructor || 'Unknown')}</span>
                <span>Lessons: ${course.lesson_count || 0}</span>
            </div>
            ${course.course_link ? `<a href="${escapeHtml(course.course_link)}" target="_blank" rel="noopener">View Course</a>` : ''}
            ${renderLessonList(course.lessons)}
        </div>
    `).join('');
}

function renderLessonList(lessons) {
    if (!lessons || lessons.length === 0) {
        return '';
    }

    return `
        <details class="lesson-list">
            <summary>View Lessons (${lessons.length})</summary>
            <ul>
                ${lessons.map(l => `
                    <li>
                        Lesson ${l.lesson_number}: ${escapeHtml(l.lesson_title || 'Untitled')}
                        ${l.lesson_link ? `<a href="${escapeHtml(l.lesson_link)}" target="_blank" rel="noopener">(link)</a>` : ''}
                    </li>
                `).join('')}
            </ul>
        </details>
    `;
}

async function scrapeCourse() {
    const url = courseUrl.value.trim();
    const cookie = sessionCookie.value.trim();

    if (!url) {
        showModal('Please provide a course URL', 'error');
        return;
    }

    if (!cookie) {
        showModal('Please provide a session cookie', 'error');
        return;
    }

    showModal('Scraping course... This may take a few minutes.', 'loading');
    scrapeBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/courses/scrape`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, session_cookie: cookie })
        });

        const data = await response.json();

        if (data.status === 'success') {
            showModal(`Successfully added "${data.course_title}" with ${data.chunk_count} chunks`, 'success');
            courseUrl.value = '';
            loadCourses();
        } else {
            showModal(data.message || 'Failed to scrape course', 'error');
        }
    } catch (error) {
        showModal('Failed to scrape course: ' + error.message, 'error');
    } finally {
        scrapeBtn.disabled = false;
    }
}

async function uploadCourse() {
    const content = courseContent.value.trim();

    if (!content) {
        showModal('Please paste course content', 'error');
        return;
    }

    // Basic validation
    if (!content.includes('Course Title:')) {
        showModal('Content must include "Course Title:" header', 'error');
        return;
    }

    showModal('Processing course...', 'loading');
    uploadBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/courses/upload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });

        const data = await response.json();

        if (data.status === 'success') {
            showModal(`Successfully added "${data.course_title}" with ${data.chunk_count} chunks`, 'success');
            courseContent.value = '';
            loadCourses();
        } else {
            showModal(data.message || 'Failed to upload course', 'error');
        }
    } catch (error) {
        showModal('Failed to upload course: ' + error.message, 'error');
    } finally {
        uploadBtn.disabled = false;
    }
}

async function deleteCourse(title) {
    if (!confirm(`Are you sure you want to delete "${title}"?`)) {
        return;
    }

    showModal('Deleting course...', 'loading');

    try {
        const response = await fetch(`${API_URL}/courses/${encodeURIComponent(title)}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.status === 'success') {
            showModal('Course deleted successfully', 'success');
            loadCourses();
        } else {
            showModal(data.message || 'Failed to delete course', 'error');
        }
    } catch (error) {
        showModal('Failed to delete course: ' + error.message, 'error');
    }
}

function showModal(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = type;
    statusModal.classList.remove('hidden');

    if (type === 'loading') {
        loadingSpinner.classList.remove('hidden');
        closeModalBtn.classList.add('hidden');
    } else {
        loadingSpinner.classList.add('hidden');
        closeModalBtn.classList.remove('hidden');
    }
}

function hideModal() {
    statusModal.classList.add('hidden');
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Make deleteCourse globally accessible for onclick handlers
window.deleteCourse = deleteCourse;
