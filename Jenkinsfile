pipeline {
    agent any
    environment {
        // --- CẤU HÌNH AWS ---
        AWS_REGION = 'us-east-1'
        // AMI Ubuntu 22.04 LTS (Deep Learning Base AMI thì bạn thay ID khác)
        EC2_AMI_ID = 'ami-0e4060c00953cd8bf'
        EC2_INSTANCE_TYPE = 'g4dn.xlarge'
        // test
        // Key Pair name ĐÃ TẠO TRÊN AWS CONSOLE
        EC2_KEY_NAME = 'test_gpu' 
        // ID của Security Group (phải mở port 22)
        EC2_SG_ID = 'sg-03dc5fdd0e2aac455' 
        PATH = "/var/jenkins_home/aws-cli-bin:${env.PATH}"
        // ID Credential lưu trong Jenkins (chứa file PEM)
        JENKINS_SSH_CRED_ID = 'ssh-eks-key' 
        AWS_CRED_ID = 'aws-credentials'
        DOCKER_HUB_CREDS = 'docker-hub-creds'
        GITHUB_CRED_ID = 'github-creds-id'
        SHORT_SHA = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()
        IMAGE_TAG = "v${env.BUILD_NUMBER}-${SHORT_SHA}" 
        DOCKER_REPO = "ne1kos0/weather-tcn-api"
        
    }

    stages {
        stage('1. Launch EC2 Instance') {
            steps {
                // BƯỚC QUAN TRỌNG: Load AWS Key vào biến môi trường
                withCredentials([usernamePassword(credentialsId: AWS_CRED_ID, passwordVariable: 'AWS_SECRET_ACCESS_KEY', usernameVariable: 'AWS_ACCESS_KEY_ID')]) {
                    script {
                        echo "Launching EC2 Instance..."
                        
                        
                        
                        def output = sh(returnStdout: true, script: """
                            aws ec2 run-instances \
                                --image-id ${EC2_AMI_ID} \
                                --count 1 \
                                --instance-type ${EC2_INSTANCE_TYPE} \
                                --key-name ${EC2_KEY_NAME} \
                                --security-group-ids ${EC2_SG_ID} \
                                --region ${AWS_REGION} \
                                --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
                                --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Jenkins-Training-Worker}]' \
                                --query 'Instances[0].InstanceId' \
                                --output text
                        """).trim()
                        
                        env.INSTANCE_ID = output
                        echo "Instance Created: ${env.INSTANCE_ID}"
                    }
                }
            }
        }

        stage('2. Wait for IP & SSH Ready') {
            steps {
                withCredentials([usernamePassword(credentialsId: AWS_CRED_ID, passwordVariable: 'AWS_SECRET_ACCESS_KEY', usernameVariable: 'AWS_ACCESS_KEY_ID')]) {
                    script {
                        echo "Waiting for Instance to be RUNNING..."
                    
                        sleep 30
                        env.INSTANCE_IP = sh(returnStdout: true, script: """
                            aws ec2 describe-instances \
                                --instance-ids ${env.INSTANCE_ID} \
                                --region ${AWS_REGION} \
                                --query 'Reservations[0].Instances[0].PublicIpAddress' \
                                --output text
                        """).trim()
                        
                        echo "Public IP: ${env.INSTANCE_IP}"
                        
                        
                        echo " Sleeping 60s for SSH Daemon to start..."
                        sleep 60
                    }
                }
            }
        }
        stage('3. SSH - Setup for Training [phase 1]') {
            steps {
                
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                        echo "🔌 Connecting via SSH..."
                        
                        
                        def remoteCommand = """
                            echo '--- FROM EC2 G4DN ---'
                            hostname
                            whoami
                            echo '--- SYSTEM SETUP ---'
                            sudo apt update
                            sudo apt install net-tools
                            sudo apt install python3-pip -y
                            sudo apt install python-is-python3 -y
                            git clone https://github.com/DarkinSideNet/test_jenkins.git -b tcn_phase
                            curl https://dl.min.io/client/mc/release/linux-amd64/mc --output mcli
                            sudo chmod +x mcli
                            sudo mv mcli /usr/local/bin/mcli
                            cd test_jenkins
                            pip install -r requirements.txt
                            echo '--- DONE ---'
                        """

                     
                         sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} \"${remoteCommand}\""

                    }
                }
            }
        }
        

        stage('4. SSH - Incremental Training [phase 1]') {
            steps {
             
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                        echo "🔌 Connecting via SSH..."
                        
                       
                        
                        def remoteCommand = """
                            echo '--- PHASE 1 TRAINING ---'
                            cd test_jenkins
                            python3 setup_minio.py
                            python3 train_incremental_2.py
                            echo '--- DONE ---'
                        """

                       
                        sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} \"${remoteCommand}\""
                        
                    }
                }
            }
        }
        
        stage('5. Evaluation & Upload [phase 2]') {
            steps {
              
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                        echo "🔌 Connecting via SSH..."
                        
                        
                        
                        def remoteCommand = """
                            echo '--- STARTING PHASE 2 EVALUATION ---'
                            cd test_jenkins
                            python3 run_evaluation.py
                            python3 ./upload_minio.py
                            echo '--- DONE ---'
                        """

                        
                        sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} \"${remoteCommand}\""
                        
                    }
                }
            }
        }

        stage('Deploy to Docker Hub') {
            steps {
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                        
                        echo "📦 Generated Tag: ${IMAGE_TAG}"
                       
                        withCredentials([usernamePassword(credentialsId: DOCKER_HUB_CREDS, usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                           
                            def remoteCommand = """
                                echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                                git clone https://github.com/DarkinSideNet/FastApi_dev.git
                                cp test_jenkins/best_model_final/weather_model_production.pth FastApi_dev/model.pth
                                cd FastApi_dev/
                                docker build -t ne1kos0/weather-tcn-api:${IMAGE_TAG} .
                                docker push ne1kos0/weather-tcn-api:${IMAGE_TAG}

                                

                            """
                            sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} 'DOCKER_USER=$DOCKER_USER DOCKER_PASS=$DOCKER_PASS bash -s' << 'EOF'\n${remoteCommand}\nEOF"
                        }
                        
                    }
                }
            }
        }
        stage('Deploy to git Hub') {
            steps {
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                       
                        echo "Deploying with Tag: ${IMAGE_TAG}"
                    
                        withCredentials([usernamePassword(credentialsId: GITHUB_CRED_ID, 
                                        usernameVariable: 'GIT_USER', 
                                        passwordVariable: 'GIT_TOKEN')]) {
                        
                            
                            def remoteCommand = """
                                set -e
                                # 1. Clean up the old folder to make room for the new copy.
                                rm -rf ~/DevOps_Projects
                                
                                # 2. Configure Git identity
                                git config --global user.email "jenkins@neikoscloud.net"
                                git config --global user.name "Jenkins CI/CD"
                                
                                # 3. Clone repo
                                git clone https://github.com/DarkinSideNet/DevOps_Projects.git ~/DevOps_Projects
                                cd ~/DevOps_Projects/charts/fastapi-ml/
                                
                                # 4. Update values-prod.yaml with the new image tag (${IMAGE_TAG})
                                sed -i 's/tag: .*/tag: "${IMAGE_TAG}"/' values-prod.yaml
                                
                                # 5. Configure Remote URL with Token to avoid password prompt when pushing
                                git remote set-url origin https://${GIT_USER}:${GIT_TOKEN}@github.com/DarkinSideNet/DevOps_Projects.git
                                
                                # 6. Commit and Push
                                git add values-prod.yaml
                                git commit -m "image-updater: update to ${IMAGE_TAG}"
                                git push origin main
                            """
                            
                         
                            sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} 'GIT_USER=$GIT_USER GIT_TOKEN=$GIT_TOKEN bash -s' << 'EOF'\n${remoteCommand}\nEOF"
                        }
                    }
                }
            }
        }
    
    }


   
    post {
        always {
            script {
              
                if (env.INSTANCE_ID) {
                    echo "TERMINATING INSTANCE ${env.INSTANCE_ID}..."
                  
                    withCredentials([usernamePassword(credentialsId: AWS_CRED_ID, passwordVariable: 'AWS_SECRET_ACCESS_KEY', usernameVariable: 'AWS_ACCESS_KEY_ID')]) {
                        sh "aws ec2 terminate-instances --instance-ids ${env.INSTANCE_ID} --region ${AWS_REGION}"
                    }
                    echo " Instance terminated."
                }
            }
        }
        failure {
            echo " Pipeline Failed! Check logs."
        }
    }
}