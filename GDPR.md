# GDPR to-do

What we still need to do before real students use the chatbot. The users are students and staff
at EU universities (sign-in is limited to `uu.se` and `student.uu.se` in
[`infra/live/aws/env.hcl`](infra/live/aws/env.hcl)), so the GDPR applies to everything we store
about them. This is a checklist, not legal advice: confirm the final setup with the university's
data protection officer.

## Personal data we handle today

| Data | Where | Why |
| --- | --- | --- |
| Email address | Cognito user pool; SES sends the sign-in code to it | Sign-in |
| Cognito user ID (`sub`) | Chat Lambda logs when a Bedrock call fails | Debugging |
| IP address, user agent | API Gateway access logs, CloudFront | Operations |
| Chat messages and answers | Sent to Bedrock; not stored by us | Answering the question |
| Uploaded course documents (planned) | S3 and the Bedrock knowledge base | RAG |

## Legal basis and roles

- [ ] Agree with the university who is the **controller** (probably the university) and who is the
      **processor** (us), and sign a data processing agreement.
- [ ] Choose and write down the legal basis for each purpose in the table above (likely public
      task or legitimate interest for sign-in and operations).
- [ ] Write a **record of processing activities** (Article 30) from that table.
- [ ] Decide whether a **DPIA** is needed. Answers from a language model to students may count as
      new technology on a large scale; ask the data protection officer.

## Where the data lives

- [ ] Move the AWS deployment to an EU region. `region` in
      [`infra/live/aws/env.hcl`](infra/live/aws/env.hcl) is `us-east-2` today; `eu-north-1`
      (Stockholm) is commented out next to it. Cognito, SES, Lambda, S3 and logs must all move.
- [ ] Keep prompts in the EU. On bedrock-mantle (the default) the endpoint is regional
      (`bedrock-mantle.<region>.api.aws`), so check that the chosen model is offered there in the EU
      region. With Converse, use an **EU** cross-region inference profile (`eu.` prefix), not a `us.`
      profile.
- [ ] Check that the AWS Data Processing Addendum (part of the AWS service terms) covers the
      account, and list AWS as a sub-processor.
- [ ] Confirm that Bedrock does not store or train on prompts and answers for the chosen model,
      and keep model invocation logging off (or, if turned on, treat it as personal data).
- [ ] Set the bedrock-mantle **data retention** mode to `none`. `GET /v1/models` shows each
      model's mode (`default` today) and the allowed ones, which include `provider_data_share`.
      The chat uses Chat Completions, which stores nothing; the Responses API would store
      conversations for 30 days by default.

## Minimise and limit retention

- [ ] Keep `log_retention_days` (14 days by default in the `auth` and `chat-api` modules) as short
      as we can and write the chosen value in the privacy notice.
- [ ] Check the API Gateway access log format: drop or truncate the IP address if we do not need it.
- [ ] Never log chat messages, answers or email addresses; add a test that the Lambda error logs
      in [`infra/modules/chat-api/lambda/chat.py`](infra/modules/chat-api/lambda/chat.py) contain only
      the user ID and the error.
- [ ] Decide how long a Cognito account lives without use, and delete inactive accounts
      automatically (for example, a scheduled Lambda).
- [ ] If we ever store chat history, set a retention period and an S3 lifecycle rule first.
- [ ] Make sure the course documents we index contain no personal data of students (grades,
      names in examples), or handle them as personal data too.

## Rights of the people using it

- [ ] **Information** (Articles 13 and 14): a privacy notice linked from the sign-in page in
      [`web/`](web) that says what we collect, why, for how long, who processes it and how to
      contact us and the data protection officer.
- [ ] **Access and portability**: a way to export what we hold about one user (the Cognito
      attributes, and any logs that mention their user ID).
- [ ] **Erasure**: a way to delete one user's account and data on request, and a short runbook
      for doing it.
- [ ] **Objection and rectification**: a contact address and a response time (one month).
- [ ] Tell users in the interface that answers come from a language model and may be wrong.

## Security

- [ ] Check that encryption at rest is on everywhere (S3, CloudWatch Logs, Cognito) and that
      CloudFront allows only HTTPS.
- [ ] Review IAM roles for least privilege; only the chat Lambda may call Bedrock.
- [ ] Limit who on the team can read CloudWatch logs and the Cognito user pool in the real account.
- [ ] Turn on CloudTrail so access to personal data is auditable.
- [ ] Write a **data breach procedure**: who to tell, and the 72-hour deadline to notify the
      supervisory authority (IMY in Sweden).

## Development and testing

- [ ] Use only made-up email addresses and documents in LocalStack, tests and CI.
- [ ] Keep `.env` files and real user data out of git and out of AI tools (Claude, Copilot).
- [ ] Add a GDPR check to the pull request review for changes that add new data, logs or regions.
