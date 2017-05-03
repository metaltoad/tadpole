'use strict';

const config = require('./config.json');
const googleapis = require('googleapis');

// Get a reference to the Knowledge Graph Search component
// const kgsearch = googleapis.kgsearch('v1');

/**
 * @param {string} user The user who sent the message.
 * @returns {object} Formatted message.
 */
function formatSlackMessage (text) {
  let entity;

  // Prepare a rich Slack message
  // See https://api.slack.com/docs/message-formatting
  const slackMessage = {
    response_type: 'in_channel',
    text: `Hello! ${text}`,
    attachments: []
  };
  return slackMessage;
}

/**
 * Verify that the webhook request came from Slack.
 *
 * @param {object} body The body of the request.
 * @param {string} body.token The Slack token to be verified.
 */
function verifyWebhook (body) {
  if (!body || body.token !== config.SLACK_TOKEN) {
    const error = new Error('Invalid credentials');
    error.code = 401;
    throw error;
  }
}

/**
 * Receive a Slash Command request from Slack.
 *
 * Trigger this function by making a POST request with a payload to:
 * https://[YOUR_REGION].[YOUR_PROJECT_ID].cloudfunctions.net/tadpole
 *
 * @example
 * curl -X POST "https://us-central1.your-project-id.cloudfunctions.net/tadpole" --data '{"token":"[YOUR_SLACK_TOKEN]","text":"giraffe"}'
 *
 * @param {object} req Cloud Function request object.
 * @param {object} req.body The request payload.
 * @param {string} req.body.token Slack's verification token.
 * @param {string} req.body.text The user's search query.
 * @param {object} res Cloud Function response object.
 */
exports.respond = function respond (req, res) {
  return Promise.resolve()
    .then(() => {
      if (req.method !== 'POST') {
        const error = new Error('Only POST requests are accepted');
        error.code = 405;
        throw error;
      }

      // Verify that this request came from Slack
      verifyWebhook(req.body);

      // Make the request to the Knowledge Graph Search API
      return formatSlackMessage(req.body.text);
    })
    .then((response) => {
      // Send the formatted message back to Slack
      res.json(response);
    })
    .catch((err) => {
      console.error(err);
      res.status(err.code || 500).send(err);
      return Promise.reject(err);
    });
};
