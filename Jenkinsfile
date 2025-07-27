pipeline {
    agent {
        // Use your slave node by label or name
        label 'slave-node'
    }

    environment {
        GIT_CREDENTIALS_ID = 'JenkinsTest'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main',
                    credentialsId: "${GIT_CREDENTIALS_ID}",
                    url: 'https://github.com/Aman-Khan/DevOps'
            }
        }

        stage('Deploy Webpage') {
            steps {
                // Copy index.html to web server root
                sh '''
                cp WebPage/index.html /var/www/html/index.html
                sudo systemctl restart apache2
                '''
            }
        }
    }
    post {
        always {
            echo "Pipeline completed — sending notification..."
        }

        success {
            emailext(
                subject: "SUCCESS: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: "Good news! Job '${env.JOB_NAME} [#${env.BUILD_NUMBER}]' ran successfully.\n\nCheck console output at ${env.BUILD_URL}",
                to: 'ak72244222@gmail.com'
            )
        }

        failure {
            emailext(
                subject: "FAILURE: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: "Oops! Job '${env.JOB_NAME} [#${env.BUILD_NUMBER}]' failed.\n\nCheck console output at ${env.BUILD_URL}",
                to: 'ak72244222@gmail.com'
            )
        }
    }
}
