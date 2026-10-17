"""Deployment of the chat on AWS: request path (solid) and sign-in (dashed).

Run: python aws_architecture.py
Writes report/figures/aws_architecture.pdf.
"""

from diagram_style import draw_arrow, draw_box, draw_group, draw_label, draw_path, new_figure, save

WIDTH, HEIGHT = 2.3, 1.3
USERS_COLUMN, CDN_COLUMN, API_COLUMN, LAMBDA_COLUMN, MODEL_COLUMN = 0.8, 4.2, 7.6, 11.0, 14.4
SIGN_IN_ROW, REQUEST_ROW, STORAGE_ROW = 7.4, 4.5, 2.3

figure, axes = new_figure(16.2, 10.0)

# Units inside the AWS account (drawn first, behind the boxes)
draw_group(axes, 2.85, 1.45, 5.55, 5.6, "web")
draw_group(axes, 6.25, 3.6, 12.35, 5.6, "chat API")
draw_group(axes, 2.85, 6.5, 12.35, 8.45, "auth")
draw_group(axes, 13.05, 1.45, 15.75, 3.4, "budget")
draw_group(axes, 2.6, 1.2, 15.95, 9.0, "AWS account, region us-east-2", filled=False)

# Request path
users = draw_box(axes, USERS_COLUMN, REQUEST_ROW, 1.4, HEIGHT, "Users", "browser")
cloudfront = draw_box(axes, CDN_COLUMN, REQUEST_ROW, WIDTH, HEIGHT, "CloudFront", "CDN, HTTPS")
bucket = draw_box(axes, CDN_COLUMN, STORAGE_ROW, WIDTH, HEIGHT, "S3 bucket", "static web page")
gateway = draw_box(axes, API_COLUMN, REQUEST_ROW, WIDTH, HEIGHT, "API Gateway", "HTTP API,\nJWT authorizer")
chat = draw_box(axes, LAMBDA_COLUMN, REQUEST_ROW, WIDTH, HEIGHT, "Lambda", "chat handler")
bedrock = draw_box(axes, MODEL_COLUMN, REQUEST_ROW, WIDTH, HEIGHT, "Bedrock", "Amazon Nova Micro", kind="llm")
draw_box(axes, MODEL_COLUMN, STORAGE_ROW, WIDTH, HEIGHT, "AWS Budgets", "e-mail alerts at\n80 % and 100 %")

# Sign-in
ses = draw_box(axes, CDN_COLUMN, SIGN_IN_ROW, WIDTH, HEIGHT, "SES", "sends the codes")
cognito = draw_box(axes, API_COLUMN, SIGN_IN_ROW, WIDTH, HEIGHT, "Cognito", "user pool,\ne-mail code sign-in")
pre_sign_up = draw_box(axes, LAMBDA_COLUMN, SIGN_IN_ROW, WIDTH, HEIGHT, "Lambda",
                       "pre sign-up check:\nuniversity domains")

# Chat requests
draw_arrow(axes, users.right, cloudfront.left, "HTTPS")
draw_arrow(axes, cloudfront.bottom, bucket.top, "static\npage", label_position="left")
draw_arrow(axes, cloudfront.right, gateway.left, "/api/*")
draw_arrow(axes, gateway.right, chat.left, "invokes")
draw_arrow(axes, chat.right, bedrock.left, "Converse")
draw_arrow(axes, gateway.top, cognito.bottom, "checks JWT", label_position="right")

# Sign-in flow
draw_arrow(axes, cognito.right, pre_sign_up.left, "pre\nsign-up")
draw_arrow(axes, cognito.left, ses.right, "sign-in\ncodes")
sign_in_x, code_x = users.x - 0.3, users.x + 0.3
draw_path(axes, [(sign_in_x, users.top[1]), (sign_in_x, 9.4), (cognito.x, 9.4), cognito.top],
          "sign in", label_at=(5.2, 9.45))
draw_path(axes, [ses.left, (code_x, ses.y), (code_x, users.top[1])],
          "e-mails\nthe code", label_at=(code_x + 0.12, 6.3), label_position="right")

draw_label(axes, 9.27, 1.05, "Provisioned as code with OpenTofu and Terragrunt; "
           "the same stack runs on LocalStack for local tests.", "below", 0)

save(figure, "aws_architecture")
