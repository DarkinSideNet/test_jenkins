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
        
    }

    stages {
        stage('1. Launch EC2 Instance') {
            steps {
                // BƯỚC QUAN TRỌNG: Load AWS Key vào biến môi trường
                withCredentials([usernamePassword(credentialsId: AWS_CRED_ID, passwordVariable: 'AWS_SECRET_ACCESS_KEY', usernameVariable: 'AWS_ACCESS_KEY_ID')]) {
                    script {
                        echo "Launching EC2 Instance..."
                        
                        
                        // Lúc này biến môi trường AWS_ACCESS_KEY_ID đã có giá trị
                        // Lệnh aws cli sẽ tự động nhận diện nó.
                        def output = sh(returnStdout: true, script: """
                            aws ec2 run-instances \
                                --image-id ${EC2_AMI_ID} \
                                --count 1 \
                                --instance-type ${EC2_INSTANCE_TYPE} \
                                --key-name ${EC2_KEY_NAME} \
                                --security-group-ids ${EC2_SG_ID} \
                                --region ${AWS_REGION} \
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
                        // sh "aws ec2 wait instance-running --instance-ids ${env.INSTANCE_ID} --region ${AWS_REGION}"
                        // sh "aws ec2 wait instance-running --instance-ids i-086cfaeaee6bcde83 --region us-east-1"
                        //Lấy Public IP
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
                // Load file PEM từ Jenkins Credential vào biến file
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                        echo "🔌 Connecting via SSH..."
                        //test
                        // Cấu hình SSH: 
                        // -o StrictHostKeyChecking=no: Để không hỏi Yes/No khi connect lần đầu
                        // ubuntu@${INSTANCE_IP}: User mặc định của AMI Ubuntu
                        
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

                        // Thực thi lệnh từ xa
                         sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} \"${remoteCommand}\""

                    }
                }
            }
        }
        

        stage('4. SSH - Incremental Training [phase 1]') {
            steps {
                // Load file PEM từ Jenkins Credential vào biến file
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                        echo "🔌 Connecting via SSH..."
                        
                        // Cấu hình SSH: 
                        // -o StrictHostKeyChecking=no: Để không hỏi Yes/No khi connect lần đầu
                        // ubuntu@${INSTANCE_IP}: User mặc định của AMI Ubuntu
                        
                        def remoteCommand = """
                            echo '--- PHASE 1 TRAINING ---'
                            cd test_jenkins
                            python3 setup_minio.py
                            python3 train_incremental_2.py
                            echo '--- DONE ---'
                        """

                        // Thực thi lệnh từ xa
                        sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} \"${remoteCommand}\""
                        
                    }
                }
            }
        }
        
        stage('5. Evaluation & Upload [phase 2]') {
            steps {
                // Load file PEM từ Jenkins Credential vào biến file
                sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
                    script {
                        echo "🔌 Connecting via SSH..."
                        
                        // Cấu hình SSH: 
                        // -o StrictHostKeyChecking=no: Để không hỏi Yes/No khi connect lần đầu
                        // ubuntu@${INSTANCE_IP}: User mặc định của AMI Ubuntu
                        
                        def remoteCommand = """
                            echo '--- STARTING PHASE 2 EVALUATION ---'
                            cd test_jenkins
                            python3 run_evaluation.py
                            python3 ./upload_minio.py
                            echo '--- DONE ---'
                        """

                        // Thực thi lệnh từ xa
                        sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} \"${remoteCommand}\""
                        
                    }
                }
            }
        }
    }


    //hối này LUÔN LUÔN chạy dù các bước trên có lỗi hay không
    post {
        always {
            script {
                // Kiểm tra nếu biến INSTANCE_ID có giá trị thì mới xóa
                if (env.INSTANCE_ID) {
                    echo "TERMINATING INSTANCE ${env.INSTANCE_ID}..."
                    // Phải dùng credentials ở đây để có quyền Admin xóa máy
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